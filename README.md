# 📄 PDF → JSON Extraction Pipeline

## Overview

This project extracts structured information from PDF documents using
layout-aware processing and configurable regex matching.

It is designed for semi-structured documents such as:

-   Tables\
-   Invoices\
-   Forms\
-   Reports

------------------------------------------------------------------------

## 🔄 Pipeline Overview

    PDF → Words → Lines → Blocks → Regex Matching → JSON

------------------------------------------------------------------------

## ⚙️ Configuration

Each run is controlled via a `.ini` file.

### Example

    [BASE]
    PATH = ./input.pdf
    DX = 10
    DY = 5

    [REGEX]
    invoice_number = \b\d{5,}\b
    date = \d{4}-\d{2}-\d{2}
    total = \b\d+\.\d{2}\b

------------------------------------------------------------------------

## 🧩 Pipeline Steps

### 1. Extract Words

Words are extracted using `pdfplumber`:

    page.extract_words(x_tolerance=2, y_tolerance=3, use_text_flow=True)

Each word becomes a bounding box.

------------------------------------------------------------------------

### 2. Merge Words → Lines

Words are grouped into lines based on:

-   Vertical alignment (`y_tol`)
-   Vertical overlap
-   Horizontal distance (`dx`)

------------------------------------------------------------------------

### 3. Merge Lines → Blocks

Lines are grouped into blocks based on:

-   Vertical gap (`dy`)
-   Horizontal distance (`dx`)

------------------------------------------------------------------------

### 4. Regex Matching

Regex patterns are applied to each line:

    match = pattern.search(line["text"])

Each match includes:

-   matched text\
-   regex groups\
-   line context\
-   block context

------------------------------------------------------------------------

### 5. JSON Output

Example output:

    {
      "path": "input.pdf",
      "hits": {
        "invoice_number": {
          "regex": "\\b\\d{5,}\\b",
          "matches": [
            {
              "match_text": "12345",
              "line": {...},
              "block": {...}
            }
          ]
        }
      }
    }

------------------------------------------------------------------------

## 🚀 Execution

Run all configs in parallel:

    python main.py

Each config generates:

    runs/<config_name>.json

------------------------------------------------------------------------

## 🧠 Design Decisions

-   Matching is done on **lines** (not words or full blocks)
-   Blocks provide **context**
-   Geometry-based grouping ensures robustness

------------------------------------------------------------------------

## ⚠️ Limitations

-   Requires text-based PDFs (not scanned images)
-   No advanced table parsing (yet)

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Column detection\
-   OCR fallback\
-   Performance optimization
