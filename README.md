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
| 4 | Data cleaning pt. 2 — types, renaming, formatting | ✅ Done |
| 5 | EDA — distributions, basic charts | ✅ Done |
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
- **Final dtypes:** 29 category, 13 int64, 6 object
- **Target:** `readmitted` — whether a patient was readmitted 
  (<30 days, >30 days, or No)
- **Class distribution:**
  - NO: 54,864
  - >30: 35,545
  - <30: 11,357

---

## 🔍 Key Findings
*(Updated as analysis progresses)*

- Dataset is imbalanced — majority of patients were NOT readmitted (54,864 NO
  vs 11,357 readmitted within 30 days)
- `[70-80)` age group has the highest hospital encounters and readmissions
- Most patients stay ~3 days (right skewed distribution)
- Female patients have higher raw readmission counts than male
- Average patient is on ~14 medications suggesting complex health conditions
- Average patient has ~50 lab procedures per stay
- Class imbalance in `<30` readmissions present across every age group —
  will need to address during modeling (SMOTE or class weighting)


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

**Day 4:**
- Loaded cleaned dataset from `data/processed/diabetes_cleaned.csv`
- Converted 29 low-cardinality columns to `category` type to save memory
  and improve performance
- Renamed columns for consistency:
  - `A1Cresult` → `a1c_result`
  - `patient_nbr` → `patient_number`
  - `diabetesMed` → `diabetes_med`
  - All hyphenated medication columns converted to underscores
    (hyphens cause errors with Python dot notation)
- Saved final cleaned dataset back to `data/processed/`

**Mistakes & Corrections:**
- `df.dtypes.value_counts()` returned fragmented category counts instead
  of a clean summary — switched to `df.dtypes.astype(str).value_counts()`
  to group dtypes correctly before counting
- Initially missed `diabetesMed` in the rename — caught it after reviewing
  the full column list and standardized to `diabetes_med`

**Day 5:**
- Created `02_eda.ipynb` and loaded cleaned dataset
- Built 6 exploratory charts:
  - Readmission distribution — confirmed class imbalance (NO >> >30 >> <30)
  - Age distribution — `[70-80)` is the dominant age group
  - Time in hospital — right skewed, most patients stay ~3 days
  - Readmission by age — older patients (50-90) dominate all readmission classes
  - Readmission by gender — females have higher raw readmission counts
  - Number of medications — peaks around 14 per patient
  - Number of lab procedures — peaks around 50 per patient
- All charts saved to `outputs/charts/`

**Mistakes & Corrections:**
- I made a simple mistake when using `seaborn`. FutureWarning appeared on histogram charts due to compatibility
  issue between installed seaborn and pandas versions. I fixed it by upgrading
  seaborn to latest version via `pip install --upgrade seaborn`, then
  restarting the Jupyter kernel and rerunning all cells
- Age distribution chart initially ordered by frequency instead of natural
  age sequence — fixed by explicitly defining `age_order` list to force
  chronological ordering `[0-10)` through `[90-100)`, which is more
  clinically meaningful and accurate
- Misidentified time in hospital distribution as left skewed — corrected
  to right skewed after reviewing: bulk of patients cluster at 1-4 days
  with a long tail extending toward longer stays

---

## ▶️ How to Run
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/healthcare-readmission-analysis.git

# Install dependencies
pip install -r requirements.txt

# Open notebooks
jupyter notebook