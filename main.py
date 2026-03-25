import configparser
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber


@dataclass
class Config:
    Path: str
    aliasDictionary: dict
    dx: int
    dy: int


@dataclass
class WordBox:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        """
        Return the width of the word box.

        Returns
        -------
        float
            Horizontal size of the box.
        """
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """
        Return the height of the word box.

        Returns
        -------
        float
            Vertical size of the box.
        """
        return self.y1 - self.y0

    @property
    def cy(self) -> float:
        """
        Return the vertical center of the word box.

        Returns
        -------
        float
            Y-coordinate of the box center.
        """
        return (self.y0 + self.y1) / 2


def load_config(config_file_path: str | Path) -> Config:
    """
    Load extraction settings from an INI config file.

    Parameters
    ----------
    config_file_path : str | Path
        Path to the INI file containing BASE and optional REGEX sections.

    Returns
    -------
    Config
        Parsed configuration object containing PDF path, regex aliases,
        horizontal tolerance, and vertical tolerance.
    """
    parser = configparser.ConfigParser()
    parser.read(config_file_path)

    path = parser["BASE"].get("PATH", "./")
    dx = int(parser["BASE"].get("DX", "5"))
    dy = int(parser["BASE"].get("DY", "5"))

    alias_dict = {}
    if "REGEX" in parser:
        alias_dict = dict(parser["REGEX"])

    return Config(Path=path, aliasDictionary=alias_dict, dx=dx, dy=dy)


def extract_words_from_pdf(cfg: Config) -> pd.DataFrame:
    """
    Extract word-level bounding boxes from a PDF file.

    Parameters
    ----------
    cfg : Config
        Extraction configuration. The PDF file is read from cfg.Path.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per extracted word and the columns:
        text, x0, y0, x1, y1
    """
    with pdfplumber.open(Path(cfg.Path)) as pdf:
        boxes = []

        for page in pdf.pages:
            all_words = page.extract_words(
                x_tolerance=2,
                y_tolerance=3,
                use_text_flow=True,
            )

            if all_words:
                for word in all_words:
                    boxes.append(
                        WordBox(
                            text=word["text"],
                            x0=word["x0"],
                            y0=word["doctop"],
                            x1=word["x1"],
                            y1=word["doctop"] + word["height"],
                        )
                    )

    return pd.DataFrame(boxes)


def _vertical_overlap_ratio(a: WordBox, b: WordBox) -> float:
    """
    Compute the vertical overlap ratio of two word boxes.

    Parameters
    ----------
    a : WordBox
        First word box.
    b : WordBox
        Second word box.

    Returns
    -------
    float
        Vertical overlap divided by the smaller box height.
        Returns 0.0 if no overlap exists.
    """
    overlap = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    denom = min(a.height, b.height)
    return overlap / denom if denom > 0 else 0.0


def _same_line(a: WordBox, b: WordBox, y_tol: float = 3.0, min_overlap: float = 0.5) -> bool:
    """
    Decide whether two word boxes belong to the same text line.

    Parameters
    ----------
    a : WordBox
        First word box.
    b : WordBox
        Second word box.
    y_tol : float
        Maximum allowed difference between vertical centers.
    min_overlap : float
        Minimum vertical overlap ratio required when center distance
        alone is not sufficient.

    Returns
    -------
    bool
        True if the boxes are considered part of the same line,
        otherwise False.
    """
    return (
            abs(a.cy - b.cy) <= y_tol
            or _vertical_overlap_ratio(a, b) >= min_overlap
    )


def _merge_word_sequence(words: list[WordBox]) -> WordBox:
    """
    Merge a left-to-right sequence of words into a single bounding box.

    Parameters
    ----------
    words : list[WordBox]
        List of word boxes belonging to one merged line fragment.

    Returns
    -------
    WordBox
        Combined word box containing merged text and the outer bounds
        of the full sequence.
    """
    words = sorted(words, key=lambda w: w.x0)
    return WordBox(
        text=" ".join(w.text for w in words),
        x0=min(w.x0 for w in words),
        y0=min(w.y0 for w in words),
        x1=max(w.x1 for w in words),
        y1=max(w.y1 for w in words),
    )


def merge_words_into_lines(
        df: pd.DataFrame,
        dx: float = 8.0,
        y_tol: float = 3.0,
        min_overlap: float = 0.5,
) -> pd.DataFrame:
    """
    Merge word boxes into line-level boxes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns: text, x0, y0, x1, y1
    dx : float
        Maximum horizontal gap allowed between neighboring words on
        the same line.
    y_tol : float
        Maximum allowed difference between vertical centers when
        assigning words to the same line.
    min_overlap : float
        Minimum vertical overlap ratio required for two words to be
        considered on the same line.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: line_id, text, x0, y0, x1, y1
    """
    words = [WordBox(**row) for row in df.to_dict(orient="records")]

    if not words:
        return pd.DataFrame(columns=["line_id", "text", "x0", "y0", "x1", "y1"])

    words.sort(key=lambda w: (w.cy, w.x0))
    line_buckets: list[list[WordBox]] = []

    for word in words:
        placed = False

        for bucket in line_buckets:
            if _same_line(word, bucket[0], y_tol=y_tol, min_overlap=min_overlap):
                bucket.append(word)
                placed = True
                break

        if not placed:
            line_buckets.append([word])

    merged_lines = []
    line_id = 0

    for bucket in line_buckets:
        bucket.sort(key=lambda w: w.x0)
        current = [bucket[0]]

        for word in bucket[1:]:
            prev = current[-1]
            gap = word.x0 - prev.x1

            if gap <= dx and _same_line(prev, word, y_tol=y_tol, min_overlap=min_overlap):
                current.append(word)
            else:
                merged = _merge_word_sequence(current)
                merged_lines.append(
                    {
                        "line_id": line_id,
                        "text": merged.text,
                        "x0": merged.x0,
                        "y0": merged.y0,
                        "x1": merged.x1,
                        "y1": merged.y1,
                    }
                )
                line_id += 1
                current = [word]

        merged = _merge_word_sequence(current)
        merged_lines.append(
            {
                "line_id": line_id,
                "text": merged.text,
                "x0": merged.x0,
                "y0": merged.y0,
                "x1": merged.x1,
                "y1": merged.y1,
            }
        )
        line_id += 1

    lines_df = pd.DataFrame(merged_lines)
    lines_df = lines_df.sort_values(["y0", "x0"], kind="stable").reset_index(drop=True)
    lines_df["line_id"] = range(len(lines_df))

    return lines_df


def merge_lines_into_blocks(
        df: pd.DataFrame,
        dy: float = 6.0,
        dx: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge line boxes into multi-line text blocks.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns: line_id, text, x0, y0, x1, y1
    dy : float
        Maximum vertical gap allowed between consecutive lines in the
        same block.
    dx : float
        Maximum horizontal distance allowed between consecutive lines
        in the same block. Overlapping lines have distance 0.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (lines_with_block_id, blocks_df)

        lines_with_block_id columns:
            line_id, block_id, text, x0, y0, x1, y1

        blocks_df columns:
            block_id, text, x0, y0, x1, y1, lines
    """
    if df.empty:
        empty_lines = pd.DataFrame(columns=["line_id", "block_id", "text", "x0", "y0", "x1", "y1"])
        empty_blocks = pd.DataFrame(columns=["block_id", "text", "x0", "y0", "x1", "y1", "lines"])
        return empty_lines, empty_blocks

    lines = df.copy().sort_values(["y0", "x0"], kind="stable").reset_index(drop=True)

    def horizontal_distance(a: pd.Series, b: pd.Series) -> float:
        """
        Compute the horizontal distance between two line boxes.

        Parameters
        ----------
        a : pd.Series
            First line record containing x0 and x1.
        b : pd.Series
            Second line record containing x0 and x1.

        Returns
        -------
        float
            Horizontal gap between the two line boxes. Returns 0.0 if
            the boxes overlap in x-direction.
        """
        if a["x1"] < b["x0"]:
            return b["x0"] - a["x1"]
        if b["x1"] < a["x0"]:
            return a["x0"] - b["x1"]
        return 0.0

    block_ids = []
    current_block_id = 0

    prev = lines.iloc[0]
    block_ids.append(current_block_id)

    for i in range(1, len(lines)):
        curr = lines.iloc[i]
        vertical_gap = curr["y0"] - prev["y1"]
        x_dist = horizontal_distance(prev, curr)

        if vertical_gap <= dy and x_dist <= dx:
            block_ids.append(current_block_id)
        else:
            current_block_id += 1
            block_ids.append(current_block_id)

        prev = curr

    lines["block_id"] = block_ids

    blocks = []
    for block_id, group in lines.groupby("block_id"):
        group = group.sort_values(["y0", "x0"], kind="stable")
        blocks.append(
            {
                "block_id": int(block_id),
                "text": "\n".join(group["text"]),
                "x0": group["x0"].min(),
                "y0": group["y0"].min(),
                "x1": group["x1"].max(),
                "y1": group["y1"].max(),
                "lines": group[["line_id", "text", "x0", "y0", "x1", "y1"]].to_dict(orient="records"),
            }
        )

    blocks_df = pd.DataFrame(blocks).sort_values(["y0", "x0"], kind="stable").reset_index(drop=True)
    return lines, blocks_df


def run_extraction(cfg: Config) -> dict[str, Any]:
    """
    Run the full extraction pipeline for one PDF.

    Parameters
    ----------
    cfg : Config
        Extraction configuration containing the PDF path, regex aliases,
        and line/block merge tolerances.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the source path and regex hits.

        Output structure:
            {
                "path": str,
                "hits": {
                    alias_name: {
                        "regex": str,
                        "matches": [
                            {
                                "match_text": str,
                                "groups": dict | tuple,
                                "line": dict,
                                "block": dict | None,
                            }
                        ],
                    }
                }
            }
    """
    words = extract_words_from_pdf(cfg=cfg)

    lines = merge_words_into_lines(
        df=words,
        dx=cfg.dx,
        y_tol=cfg.dy,
    )

    lines, blocks = merge_lines_into_blocks(
        df=lines,
        dy=cfg.dy,
        dx=cfg.dx,
    )

    block_lookup = {
        row["block_id"]: row
        for row in blocks.to_dict(orient="records")
    }

    hits: dict[str, dict[str, Any]] = {}
    line_records = lines.to_dict(orient="records")

    for name, regex in cfg.aliasDictionary.items():
        pattern = re.compile(regex)
        matches: list[dict[str, Any]] = []

        for line in line_records:
            text = line.get("text") or ""
            match = pattern.search(text)
            if not match:
                continue

            block = block_lookup.get(line["block_id"])
            matches.append(
                {
                    "match_text": match.group(0),
                    "groups": match.groupdict() if match.re.groupindex else match.groups(),
                    "line": line,
                    "block": block,
                }
            )

        hits[name] = {
            "regex": regex,
            "matches": matches,
        }

    return {
        "path": str(cfg.Path),
        "hits": hits,
    }


def run_single_config(file: Path, runs_dir: Path) -> Path:
    """
    Run extraction for one config file and write the result as JSON.

    Parameters
    ----------
    file : Path
        Path to the input INI configuration file.
    runs_dir : Path
        Directory where the output JSON file will be written.

    Returns
    -------
    Path
        Path to the generated JSON output file.
    """
    cfg = load_config(file)
    result = run_extraction(cfg)

    output_file = runs_dir / f"{file.stem}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return output_file


if __name__ == "__main__":
    config_files = list(Path("config").glob("*.ini"))
    config_files.remove(Path("config/global.ini"))

    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(run_single_config, file, runs_dir): file
            for file in config_files
        }

        for future in as_completed(futures):
            file = futures[future]
            try:
                output_file = future.result()
                print(f"Done: {file.name} -> {output_file}")
            except Exception as e:
                print(f"Failed: {file.name}: {e}")

    print("All config runs finished.")