# 🏥 Healthcare Patient Readmission Risk & Prediction Pipeline

## Overview
An end-to-end data science and data engineering project analyzing patient readmission 
risk using the UCI Diabetes 130-US Hospitals dataset from 1998-2008. The goal is to identify key 
factors that contribute to hospital readmissions within 30 days and build a 
predictive model to flag high-risk patients.

---

## 🗂️ Project Structure

healthcare-readmission-analysis/
│
├── data/
│   ├── raw/                  # Original dataset (unchanged)
│   └── processed/            # Cleaned/ready-to-use data
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering_modeling.ipynb
│   └── 04_genai_summary.ipynb
│
├── sql/
│   └── queries.sql
│
├── src/
│   └── helper_functions.py
│
├── outputs/
│   ├── charts/
│   └── model_results/
│
├── README.md
├── requirements.txt
└── .gitignore

---

## 🛠️ Tools & Technologies
- **Python** — pandas, numpy, scikit-learn, matplotlib, seaborn
- **Jupyter Notebooks** — analysis and modeling
- **SQL** — exploratory queries
- **Git/GitHub** — version control

---

## 📅 Build Progress (14-Day Plan)

| Day | Task | Status |
|-----|------|--------|
| 1 | Repo setup, folder structure, dataset download | ✅ Done |
| 2 | Load & inspect data — shape, types, nulls, value counts | ✅ Done |
| 3 | Data cleaning pt. 1 — missing values, duplicates, errors | ✅ Done |
| 4 | Data cleaning pt. 2 — types, renaming, formatting | 🔲 |
| 5 | EDA — distributions, basic charts | 🔲 |
| 6 | Deeper EDA — feature vs readmission relationships | 🔲 |
| 7 | Feature planning — notes, ideas, no heavy coding | 🔲 |
| 8 | Feature engineering — age groups, visit counts, risk flags | 🔲 |
| 9 | First model — scikit-learn baseline | 🔲 |
| 10 | Model evaluation — accuracy, precision, recall | 🔲 |
| 11 | Model improvement — new features or different model | 🔲 |
| 12 | Insights & storytelling — key findings | 🔲 |
| 13 | GitHub + README polish | 🔲 |
| 14 | Final polish, resume update, practice walkthrough | 🔲 |

---

## 📊 Dataset
- **Source:** UCI ML Repository — Diabetes 130-US Hospitals
- **Original shape:** 101,766 rows x 50 columns
- **Cleaned shape:** 101,766 rows x 48 columns
- **Target:** `readmitted` — whether a patient was readmitted (<30 days, >30 days, or No)

---

## 🔍 Key Findings
*(To be filled in as analysis progresses)*

- TBD after EDA
- TBD after modeling

---

## 🤖 Model Results
*(To be filled in — Day 9–11)*

| Metric | Score |
|--------|-------|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 Score | TBD |

---

## 💡 Insights & Business Value
*(To be filled in — Day 12)*

---

## 🧠 Personal Notes (Dev Log)
> These are informal notes for my own reference during the build.

**Day 1:**
- Set up repo and folder structure
- Downloaded dataset from UCI repository
- Skimmed columns — 50 features, mix of categorical and numeric

**Day 2:**
- Loaded dataset with pandas
- Shape: [ADD YOUR SHAPE HERE] rows, [ADD] columns
- No duplicate rows found
- Missing values found in: [LIST COLUMNS WITH NULLS HERE]
- Target column `readmitted` value counts: [ADD COUNTS]
- First impression: [WRITE 1-2 SENTENCES ABOUT WHAT LOOKS MESSY OR INTERESTING]

**Day 3:**
- Replaced all `?` placeholders with NaN
- Filled missing values: race, medical_specialty, diag_1/2/3, max_glu_serum, A1Cresult
- Dropped columns: weight (98% missing), payer_code (not clinically relevant)
- Final shape: 101,766 rows x 48 columns
- Zero nulls remaining
- Saved cleaned dataset to `data/processed/diabetes_cleaned.csv`

---

## ▶️ How to Run
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/healthcare-readmission-analysis.git

# Install dependencies
pip install -r requirements.txt

# Open notebooks
jupyter notebook