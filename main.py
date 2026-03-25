import configparser
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pdfplumber


@dataclass
class Config:
    Path: str
    aliasDictionary: dict


@dataclass
class WordBox:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


def load_config(config_file_path) -> Config:
    parser = configparser.ConfigParser()
    parser.read(config_file_path)
    path = parser["BASE"].get("PATH", "./")

    alias_dict = {}
    if "REGEX" in parser:
        alias_dict = dict(parser["REGEX"])

    return Config(Path=path, aliasDictionary=alias_dict)


def extract_words_from_pdf(cfg: Config) -> pd.DataFrame:
    with pdfplumber.open(Path(cfg.Path)) as pdf:
        boxes = []
        for page in pdf.pages:
            all_words = page.extract_words(x_tolerance=2, y_tolerance=3, use_text_flow=True)
            # Wenn die Liste der Wörter nicht leer ist.
            if all_words:
                # Word ist ein {}
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


def axis_gap(a0: float, a1: float, b0: float, b1: float) -> float:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0


def boxes_close(a: WordBox, b: WordBox, dx: float, dy: float) -> bool:
    gap_x = axis_gap(a.x0, a.x1, b.x0, b.x1)
    gap_y = axis_gap(a.y0, a.y1, b.y0, b.y1)
    return gap_x <= dx and gap_y <= dy


def merge_component(group: list[WordBox]) -> WordBox:
    group = sorted(group, key=lambda w: (w.y0, w.x0))
    return WordBox(
        text=" ".join(w.text for w in group),
        x0=min(w.x0 for w in group),
        y0=min(w.y0 for w in group),
        x1=max(w.x1 for w in group),
        y1=max(w.y1 for w in group),
    )


def merge_words_into_blocks(df: pd.DataFrame, dx: float = 15, dy: float = 5) -> pd.DataFrame:
    words = [WordBox(**row) for row in df.to_dict(orient="records")]
    n = len(words)

    if n == 0:
        return pd.DataFrame(columns=["text", "x0", "y0", "x1", "y1"])

    parent = list(range(n))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if boxes_close(words[i], words[j], dx, dy):
                union(i, j)

    groups: dict[int, list[WordBox]] = {}
    for i, w in enumerate(words):
        groups.setdefault(find(i), []).append(w)

    merged = [merge_component(group) for group in groups.values()]
    merged.sort(key=lambda w: (w.y0, w.x0))

    return pd.DataFrame([vars(w) for w in merged])


def run_extraction(cfg: Config) -> dict:
    words = extract_words_from_pdf(cfg=cfg)
    blocks = merge_words_into_blocks(df=words, dx=15, dy=10)

    hits: dict[str, dict] = {}

    for name, regex in cfg.aliasDictionary.items():
        pattern = re.compile(regex)
        matched = blocks[blocks["text"].str.contains(pattern, na=False, regex=True)]
        hits[name] = {
            "regex": regex,
            "blocks": matched.to_dict(orient="records"),
        }

    return {
        "path": str(cfg.Path),
        "hits": hits,
    }


def run_single_config(file: Path, runs_dir: Path) -> Path:
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