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
| 6 | Deeper EDA — feature vs readmission relationships | ✅ Done |
| 7 | Feature planning — notes, ideas, no heavy coding | ✅ Done |
| 8 | Feature engineering — age groups, visit counts, risk flags | ✅ Done |
| 9 | First model — scikit-learn baseline | ✅ Done |
| 10 | Model evaluation — accuracy, precision, recall | ✅ Done |
| 11 | Improve model — new features, different model | ✅ Done |
| 12 | Insights & storytelling — key findings, business value | ✅ Done |
| 13 | GitHub + README polish | 🔲 |
| 14 | Final polish, resume update, practice walkthrough | 🔲 |

---

## 📊 Dataset
- **Source:** UCI ML Repository — Diabetes 130-US Hospitals
- **Original shape:** 101,766 rows x 50 columns
- **Cleaned shape:** 101,766 rows x 48 columns
- **Feature engineered shape:** 101,766 rows x 47 columns
- **Final dtypes:** 29 category, 13 int64, 6 object
- **Original target:** `readmitted` — NO, >30, <30
  - NO: 54,864
  - >30: 35,545
  - <30: 11,357
- **Model target:** `readmitted_binary` — 1 if readmitted within 30 days, 0 otherwise
  - Not readmitted within 30 days: 90,409 (88.84%)
  - Readmitted within 30 days: 11,357 (11.16%)
- **Readmission rate:** 11.16% — significantly imbalanced dataset

---

## 🔍 Key Findings

**Data & EDA Findings:**
- Dataset is imbalanced — majority of patients were NOT readmitted (54,864 NO
  vs 11,357 readmitted within 30 days)
- `[70-80)` age group has the highest hospital encounters and readmissions
- Most patients stay ~3 days (right skewed distribution)
- Female patients have higher raw readmission counts than male
- Average patient is on ~14 medications suggesting complex health conditions
- Average patient has ~50 lab procedures per stay
- Class imbalance in `<30` readmissions present across every age group —
  addressed with `class_weight='balanced'` in model
- Longer hospital stays correlate with higher readmission risk
- Higher medication counts correlate with higher readmission risk —
  `<30` patients average 16+ medications vs 14-16 for non-readmitted
- Higher lab procedure counts generally correlate with readmission risk
  but outliers in `NO` class suggest it is not a standalone predictor
- Caucasian patients dominate all readmission classes reflecting overall
  dataset composition — proportional analysis needed for meaningful comparison

**Feature Engineering Findings:**
- 5 new features engineered — `age_numeric`, `age_group`, `total_visits`,
  `high_risk_flag`, `readmitted_binary`
- 8,210 patients (~8%) flagged as high risk based on long stay and
  high medication count
- 50% of patients have 0 prior visits — first time or no recorded history
- Max of 80 total visits for a single patient — extreme utilizers present

**Modeling Findings:**
- Baseline Logistic Regression achieves ROC AUC of 0.6401 — better than
  random guessing, room for improvement
- Recall of 48% on readmitted class — catching nearly half of actual
  readmissions with a simple baseline model
- In healthcare context recall matters more than precision — missing a
  true readmission is more costly than a false alarm
- `number_inpatient` is the strongest predictor of readmission risk —
  prior hospital history is the biggest signal
- `number_outpatient` is a protective factor — active outpatient
  engagement reduces readmission risk
- High value use case identified: target patients with high inpatient
  history and low outpatient engagement for intervention programs
- Best model: Random Forest v2 — AUC 0.6482 after tuning
- Random Forest v1 was overfit (89% accuracy, 1% recall) — tuning with
  `max_depth=10` and `min_samples_leaf=10` fixed minority class detection

---

## 🤖 Model Results

### Model Comparison
| Model | Accuracy | ROC AUC | Recall (readmitted) | F1 (readmitted) |
|-------|----------|---------|---------------------|-----------------|
| Logistic Regression baseline | 68% | 0.6401 | 48% | 0.25 |
| Logistic Regression v2 | 68% | 0.6438 | 49% | 0.25 |
| Random Forest v1 | 89% | 0.5977 | 1% | 0.03 |
| Random Forest v2 (tuned) | 69% | **0.6482** | 48% | 0.25 |

### Best Model — Random Forest v2 (tuned)
| Metric | Score |
|--------|-------|
| Accuracy | 69% |
| ROC AUC | 0.6482 |
| Precision (readmitted) | 17% |
| Recall (readmitted) | 48% |
| F1 Score (readmitted) | 0.25 |

### Baseline — Logistic Regression
**Confusion Matrix:**
| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 12,784 (TN) | 5,299 (FP) |
| **Actual 1** | 1,172 (FN) | 1,099 (TP) |

**Feature Importance (Top Predictors):**
| Feature | Direction | Insight |
|---------|-----------|---------|
| `number_inpatient` | Positive ↑ | More prior inpatient visits = higher risk |
| `number_outpatient` | Negative ↓ | More outpatient visits = lower risk |
| `time_in_hospital` | Positive ↑ | Longer stays = higher risk |
| `num_medications` | Positive ↑ | More medications = higher risk |
| `high_risk_flag` | Positive ↑ | Combined risk flag = higher risk |

**Evaluation Charts:**
- Confusion matrix saved to `outputs/model_results/confusion_matrix.png`
- ROC curve saved to `outputs/model_results/roc_curve.png`
- Feature importance saved to `outputs/model_results/feature_importance.png`
- Best model saved to `outputs/model_results/best_model_rf_v2.pkl`

**Key Takeaways:**
- Random Forest v1 overfitting shows high accuracy alone is misleading
  on imbalanced datasets — always check recall on minority class
- Incremental AUC improvements achieved through iteration and tuning
- False negatives are the most costly error in healthcare context —
  a missed readmission means no intervention for a high risk patient
- Further gains would require SMOTE oversampling or deeper feature
  engineering on diagnosis codes

## 💡 Insights & Business Value

### The Problem
Hospital readmissions within 30 days are costly for both patients and 
healthcare systems. This project identifies which patients are at highest 
risk so hospitals can intervene proactively before discharge.

### What Drives Readmissions
- **Prior inpatient visits** are the strongest predictor — patients with
  a history of hospital admissions are significantly more likely to return
- **Longer hospital stays** (4+ days) correlate with higher readmission risk
- **Higher medication counts** (16+ medications) indicate more complex
  conditions and higher risk
- **Outpatient engagement is protective** — patients who regularly see
  doctors outside the hospital are less likely to be readmitted

### Business Value
1. **Early intervention** — Flag high risk patients before discharge for
   follow up programs based on inpatient history and medication count
2. **Resource allocation** — Focus discharge planning on patients staying
   4+ days with 16+ medications
3. **Outpatient investment** — Data supports investing in outpatient follow
   up programs as a readmission reduction strategy
4. **Cost reduction** — Even a small reduction in readmission rate has
   significant financial impact on hospital systems

### Limitations & Next Steps
- Diagnosis codes not yet used — contain rich clinical information that
  could significantly improve predictions
- Dataset from 1999-2008 — treatment patterns may have changed
- Future improvements: SMOTE oversampling, XGBoost, diagnosis code
  encoding, hyperparameter tuning with GridSearchCV

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


**Day 6:**
- Continued in `02_eda.ipynb` with deeper feature vs readmission analysis
- Built 5 deeper EDA charts:
  - Time in hospital vs readmission (boxplot)
  - Number of medications vs readmission (boxplot)
  - Number of lab procedures vs readmission (boxplot)
  - Readmission by race (countplot)
  - Average time in hospital by readmission class (aggregation barplot)
  - Average medications by readmission class (aggregation barplot)
- Key features identified as strong readmission predictors:
  - `time_in_hospital` — longer stays correlate with higher readmission risk
  - `num_medications` — more medications correlate with higher readmission risk
  - `num_lab_procedures` — higher lab counts correlate with risk but with nuance
- Noted `Unknown` category in race column — patients who did not disclose race
- All charts saved to `outputs/charts/`

**Mistakes & Corrections:**
- Got `NameError: name 'plt' is not defined` when running boxplot cell —
  caused by restarting the kernel earlier which cleared all variables from
  memory. Fixed by rerunning Cell 1 imports and Cell 2 data loading before
  proceeding. Reminder: always rerun imports and data load cells after
  a kernel restart
- Initially unsure how to interpret the lab procedures boxplot — worked
  through it systematically by breaking down median, spread, and outliers
  separately for each readmission class rather than trying to read everything
  at once

**Day 7:**
- Light planning day — no heavy coding
- Reviewed all EDA findings from Days 5 and 6
- Identified strong existing features for modeling:
  - `time_in_hospital`, `num_medications`, `num_lab_procedures`
  - `number_inpatient`, `number_emergency`, `number_outpatient`
  - `age`, `a1c_result`, `max_glu_serum`, `diabetes_med`, `change`
- Planned new features to engineer on Day 8:
  - `age_numeric` — convert age brackets to numeric midpoint values
  - `age_group` — simplified buckets (Young, Middle, Senior, Elderly)
  - `total_visits` — sum of all prior visit types
  - `high_risk_flag` — binary flag for high risk patients
  - `readmitted_binary` — simplified binary target variable
- Identified columns to drop before modeling:
  - `encounter_id`, `patient_number` — ID columns, no predictive value
  - `diag_1`, `diag_2`, `diag_3` — 700+ unique values, too complex
    for baseline model, revisit on Day 11 if time allows
  - `medical_specialty` — 73 unique values, needs grouping first

**Mistakes & Corrections:**
- No errors today — planning and documentation day only

**Day 8:**
- Created `03_feature_engineering_modeling.ipynb`
- Loaded cleaned dataset from `data/processed/diabetes_cleaned.csv`
- Engineered 5 new features:
  - `age_numeric` — converted age brackets to midpoint numeric values
    (e.g. `[70-80)` → 75) so model understands age as ordered number
  - `age_group` — simplified buckets: Young (0-40), Middle (40-60),
    Senior (60-80), Elderly (80+)
  - `total_visits` — sum of `number_inpatient + number_outpatient +
    number_emergency` to capture overall healthcare utilization
  - `high_risk_flag` — binary flag: 1 if `time_in_hospital > 7` AND
    `num_medications > 20`, combines two strongest EDA signals
  - `readmitted_binary` — binary target: 1 if readmitted within 30 days
- Dropped 6 columns: `encounter_id`, `patient_number`, `diag_1/2/3`,
  `medical_specialty`
- Final shape: 101,766 rows x 47 columns
- Readmission rate confirmed at 11.16% — significant class imbalance
- Saved model ready dataset to `data/processed/diabetes_features.csv`

**Mistakes & Corrections:**
- Made a silly mistake and used hardcoded full file path and wrong filename
  `diabetes_data.csv` — fixed by switching to relative path
  `../data/processed/diabetes_cleaned.csv`. Relative paths are portable
  and work on any machine, hardcoded paths do not

  **Day 9:**
- Built baseline Logistic Regression model using 11 features
- Selected features based on EDA findings:
  - `time_in_hospital`, `num_lab_procedures`, `num_procedures`,
    `num_medications`, `number_outpatient`, `number_emergency`,
    `number_inpatient`, `number_diagnoses`, `age_numeric`,
    `total_visits`, `high_risk_flag`
- Scaled features using `StandardScaler` — required for Logistic Regression
- Used `stratify=y` in train/test split to maintain 11.16% class balance
  in both train and test sets
- Used `class_weight='balanced'` to handle class imbalance
- Split: 80% train (81,412 samples), 20% test (20,354 samples)
- Model trained and evaluated successfully

**Baseline Results:**
- Accuracy: 68%
- ROC AUC: 0.64
- Recall (readmitted class): 48%
- Precision (readmitted class): 17%
- True Positives: 1,099 | False Negatives: 1,172
- True Negatives: 12,784 | False Positives: 5,299

**Mistakes & Corrections:**
- Noted that accuracy alone is a misleading metric for imbalanced datasets
  — a model predicting 0 for everything would score 88% accuracy but have
  zero predictive value. ROC AUC and recall are more meaningful metrics
  for this problem
- `stratify=y` used in train/test split to ensure class balance is
  maintained — without this one split could have more readmissions than
  the other which would skew results

**Day 10:**
- Visualized confusion matrix using `ConfusionMatrixDisplay`
- Plotted ROC curve — AUC 0.64 confirmed visually, curve bows above
  diagonal but stays relatively close, significant room to improve
- Explored feature importance using Logistic Regression coefficients:
  - Strongest positive predictor: `number_inpatient` — more prior
    inpatient visits = higher readmission risk
  - Strongest negative predictor: `number_outpatient` — more outpatient
    visits = lower readmission risk (acts as protective factor)
- All evaluation charts saved to `outputs/model_results/`

**Key Insight from Feature Importance:**
- Patients with high inpatient history and low outpatient engagement
  are the highest risk group — hospitals could use this to target
  intervention programs proactively

**Mistakes & Corrections:**
- No major errors today — ensured `outputs/model_results/` folder
  existed before saving charts using `os.makedirs()` with
  `exist_ok=True` to avoid FileNotFoundError

  **Day 11:**
- Expanded feature set from 11 to 16 features by adding:
  - `age_group`, `diabetes_med`, `change`
- Encoded categorical features using `pd.get_dummies`
- Trained and compared 4 models total:
  - Logistic Regression baseline — AUC 0.6401
  - Logistic Regression v2 (16 features) — AUC 0.6438
  - Random Forest v1 — AUC 0.5977 (overfit)
  - Random Forest v2 tuned — AUC 0.6482 (best)
- Best model saved to `outputs/model_results/best_model_rf_v2.pkl`

**Mistakes & Corrections:**
- Random Forest v1 achieved 89% accuracy but only 1% recall on
  readmitted class — model was essentially ignoring the minority class
  entirely despite `class_weight='balanced'`. This is a classic
  imbalanced dataset trap — high accuracy is misleading when one class
  dominates. Fixed by tuning with `max_depth=10` and
  `min_samples_leaf=10` to reduce overfitting, bringing recall back
  up to 48% and AUC to 0.6482
- Learned that accuracy alone is never sufficient for imbalanced
  datasets — always evaluate ROC AUC and recall on minority class

  **Day 12:**
- Wrote full insights and storytelling section in notebook
- Summarized findings in plain business language for non-technical readers
- Identified 4 concrete business value opportunities:
  1. Early intervention targeting based on inpatient history
  2. Resource allocation for discharge planning
  3. Outpatient program investment as protective factor
  4. Cost reduction through proactive readmission prevention
- Documented limitations:
  - Dataset from 1999-2008, treatment patterns may have changed
  - Diagnosis codes not used — rich clinical signal left on table
  - Model explains ~65% of variance, room to improve
- Documented improvement roadmap:
  - SMOTE oversampling for class imbalance
  - XGBoost or gradient boosting
  - Diagnosis code encoding into clinical categories
  - GridSearchCV hyperparameter tuning
- Added insights and business value section to project README

**Mistakes & Corrections:**
- No major errors today — insights and documentation day
- Learned that storytelling in plain language is as important as
  the technical work — recruiters read the README before the notebooks
---

## ▶️ How to Run
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/healthcare-readmission-analysis.git

# Install dependencies
pip install -r requirements.txt

# Open notebooks
jupyter notebook