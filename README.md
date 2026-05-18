# EccLexica : Christian Architectural Discourse Detection in Latin Texts

This repository contains a dictionary-based method for identifying architectural discourse in Late Antique and Medieval Latin texts. It is applied to a test corpus of Latin inscriptions.

The workflow combines a **dictionary-based approach** with an **evaluation against GLiNER named-entity recognition models**.

EccLexica is being developed as part of the ANR E-cclesia project.

<div align="center">
  <img width="100" src="assets/ANR_Logo.svg.png" alt="logo anr"/>
  <img width="100" src="assets/logo_ecclesia.jpg" alt="logo ecclesia"/>
</div>

---

## Repository Structure

```text
.
├── data/
│   └── dicts
│     └── auto_terms.csv
│     └── asso_terms.csv
│     └── mat_terms.csv
│   └── sample.csv
│
├── output/
│
├── inscr-preprocess.py
├── inscr-lemmatize.py
├── inscr-score.py
├── gliner-eval.py
└── README.md
```

---

## Data

- **Main dataset**

The main dataset is not stored in this repository due to its size.

Please download **EDCS_text_cleaned_2022-09-12.json** from Zenodo:

https://zenodo.org/records/7072337

After downloading, place the file in the following location:

data/EDCS_text_cleaned_2022-09-12.json

- **Sample dataset** (`data/sample.csv`)
  - 100 inscriptions.
  - Used for testing and GLiNER evaluation.

---

## Scripts

This repository includes three standalone command-line scripts for a split workflow:

1. `preprocess.py` — filter raw JSON inscription data by date and province.
2. `lemmatize.py` — lemmatize filtered texts with spaCy.
3. `score.py` — calculate architectural scores from lemmatized texts.

---

## Requirements

- Python 3.8+
- `pandas`
- `numpy`
- `spacy`
- `gliner`
- Latin spaCy model: `la_core_web_md`

Install dependencies if needed:

```bash
pip install pandas numpy spacy
python -m spacy download la_core_web_md
```

---

## 1. Preprocess

This step loads a JSON file, filters inscriptions by the default date range and excluded provinces, and saves the filtered CSV.

```bash
python preprocess.py \
  --input data/EDCS_text_cleaned_2022-09-12.json \
  --output data/edcs_filtered_inscriptions.csv
```

Use `--exclude-provinces` to supply a comma-separated list of provinces to exclude without an interactive prompt:

```bash
python preprocess.py --exclude-provinces "Achaia,Aegyptus,Syria" --no-prompt
```

---

## 2. Lemmatize

This step reads the filtered CSV and adds a `lemmatized_text` column.

```bash
python lemmatize.py \
  --input data/edcs_filtered_inscriptions.csv \
  --output data/edcs_lemmatized_inscriptions.csv
```

---

## 3. Score

This step reads the lemmatized CSV, calculates architectural scores, and writes the scored output.

```bash
python score.py \
  --input data/edcs_lemmatized_inscriptions.csv \
  --output output/edcs_architectural_scores.csv \
  --high-output output/edcs_architectural_scores_gt50.csv
```

To disable creation of per-score-bin subset CSV files, use:

```bash
python score.py --no-subsets
```

---

## Notes

- `preprocess.py` may prompt interactively for excluded provinces unless `--exclude-provinces` or `--no-prompt` is used.
- `lemmatize.py` uses spaCy and needs the model installed.
- `score.py` expects the lemmatized data to contain a `lemmatized_text` column.

### `gliner-eval.py`

Evaluates **GLiNER NER models** against the dictionary-based architectural terms.

**What it does:**
- Loads a scored sample dataset from `output/sample.csv`.
- Runs multiple GLiNER models on:
  - Lemmatized text
  - Cleaned interpretive text
- Compares GLiNER-extracted entities with dictionary-based terms.

**Evaluated labels:**
- `building/type`
- `building/part`
- `building/material`

**Metrics computed:**
- Precision
- Recall
- F1 score
- Percentage and count of overlapping terms
- Number of inscriptions with at least one shared term

---

## Notes

- Scores are heuristic and intended for **comparative and exploratory analysis**, not absolute classification.

---

## Citations 

Heřmánková, Petra. “EDCS”. Zenodo, September 12, 2022. https://doi.org/10.5281/zenodo.7072337.
