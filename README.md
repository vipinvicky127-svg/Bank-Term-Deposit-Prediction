# 🏦 Bank Term Deposit Prediction: Machine Learning for Telemarketing Optimization

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Project%20Status-Completed-success.svg)]()

> An end-to-end Machine Learning classification system designed to predict whether retail banking customers will subscribe to a term deposit during direct telemarketing campaigns, cutting marketing overhead and maximizing campaign ROI.

---

## 📌 Executive Summary & Business Context

Term deposits are a cornerstone funding source for retail banking institutions, providing stable, long-term liquidity with predictable interest rates. To acquire new term deposit accounts, commercial banks conduct large-scale telephonic marketing campaigns.

However, cold calling without predictive intelligence is highly inefficient:
* **High Operational Costs:** Outbound call centers incur significant agent hourly costs.
* **Low Baseline Conversion:** Only **11.74%** of targeted prospects in historical campaigns actually subscribe.
* **Customer Fatigue:** Repetitive non-targeted calls increase customer dissatisfaction and churn risk.

### The Objective
Build a predictive machine learning classification pipeline that screens client demographic profiles and interaction records to identify high-probability prospects **before and during** campaign outreach, improving conversion rates while significantly decreasing wasted call volume.

---

## 🏆 Key Performance Results

| Model | Validation Accuracy | Validation ROC-AUC | Test Accuracy (`solution_checker.xlsx`) |
|:---|:---:|:---:|:---:|
| **Logistic Regression** (Baseline) | 90.38% | 0.9088 | 90.22% |
| **Decision Tree** (Baseline depth=4) | 90.41% | 0.8506 | 90.75% |
| **Random Forest** (n=100, depth=10) | 89.83% | 0.9280 | 90.11% |
| **Gradient Boosting** (Selected) | **90.74%** | **0.9326** | **90.81%** (12,318 / 13,564) |

* The **Gradient Boosting Classifier** achieved the strongest overall discrimination with a **0.9326 ROC-AUC**, delivering **90.81% accuracy** verified against the competition ground truth in `solution_checker.xlsx`.

---

## 🔍 Key Exploratory Data Analysis (EDA) Insights

1. **Call Duration is the Strongest Predictor:**
   Call duration exhibits a correlation of +0.39 with customer conversion. Calls lasting longer than 5 minutes (300 seconds) show a steep increase in conversion rates from <8% to >45%.
2. **High-Converting Demographics (Students & Retirees):**
   While blue-collar and management roles represent the highest call volume, **students (~28%)** and **retired individuals (~25%)** exhibit the highest conversion percentages.
3. **Past Campaign Track Record Matters Most (`poutcome`):**
   Prospects who responded positively in prior marketing campaigns (`poutcome == 'success'`) convert at over **60%**, making past campaign outcome a high-priority targeting signal.
4. **Debt Burden Suppresses Conversion:**
   Clients holding an active housing loan (`housing == 'yes'`) or personal loan (`loan == 'yes'`) convert at roughly half the rate of debt-free clients.

---

## 🏗️ Project Architecture & Repository Structure

```text
Bank-Term-Deposit-Prediction/
├── .gitignore               # Ignored files (caches, virtualenvs, checkpoints)
├── LICENSE                  # Open-source MIT License
├── README.md                # Comprehensive project documentation
├── requirements.txt         # Pinned project dependencies
├── run_pipeline.py          # CLI executable to run end-to-end ML pipeline
│
├── data/                    # Dataset storage and documentation
│   ├── data_dictionary.md   # Feature descriptions and schema details
│   ├── train.csv            # 31,647 training records with target label
│   ├── test.csv             # 13,564 unlabeled test records
│   └── Problem Statement.pdf# Official business specification PDF
│
├── notebooks/               # Interactive Jupyter analysis notebooks
│   ├── Bank_Term_Deposit_Analysis_and_Modeling.ipynb  # Polished full walkthrough
│   └── Benchmark_Solution.ipynb                       # Original baseline notebook
│
├── src/                     # Modular source package
│   ├── __init__.py          # Package initialization
│   ├── data_loader.py       # Data loading and schema validation
│   ├── preprocess.py        # One-hot encoding, feature alignment, train/val split
│   ├── models.py            # Model factory and training routines
│   ├── evaluate.py          # Metrics calculation and reporting
│   └── predict.py           # Test inference and solution checker verification
│
└── outputs/                 # Model outputs and verification files
    ├── submission.csv       # Final test predictions (13,564 rows)
    └── solution_checker.xlsx# Official automated score verification workbook
```

---

## 📊 Dataset Schema & Features

The dataset comprises **31,647 training rows** and **13,564 test rows** across 17 input attributes:

### 1. Client Demographics & Financials
* `ID`: Unique client identification code (excluded from model training).
* `age`: Age in years (numeric: 18 - 95).
* `job`: Occupation type (12 categories: `admin.`, `blue-collar`, `technician`, `services`, `management`, `retired`, `student`, etc.).
* `marital`: Marital status (`married`, `single`, `divorced`).
* `education`: Education tier (`primary`, `secondary`, `tertiary`, `unknown`).
* `default`: Has credit in default? (`no`, `yes`).
* `balance`: Average yearly bank balance in Euros (numeric).
* `housing`: Has active housing loan? (`no`, `yes`).
* `loan`: Has personal loan? (`no`, `yes`).

### 2. Campaign Interaction Details
* `contact`: Communication channel (`cellular`, `telephone`, `unknown`).
* `day`: Day of the month of the last contact (1 - 31).
* `month`: Last contact month of year (`jan` - `dec`).
* `duration`: Last contact duration in seconds (numeric).
* `campaign`: Number of contacts during this campaign (numeric).
* `pdays`: Days elapsed since last contact from a previous campaign (`-1` = never contacted).
* `previous`: Number of contacts prior to this campaign (numeric).
* `poutcome`: Outcome of the previous marketing campaign (`failure`, `other`, `success`, `unknown`).

### 3. Target Variable
* `subscribed`: Did the client subscribe to a term deposit? (`yes`, `no`).

*(For full column types and value distributions, see [data_dictionary.md](data/data_dictionary.md).)*

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/vipinvicky127-svg/Bank-Term-Deposit-Prediction.git
cd Bank-Term-Deposit-Prediction
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Option 1: Execute the Automated Machine Learning Pipeline via CLI

Run the full pipeline (load data ➔ preprocess ➔ train Gradient Boosting ➔ evaluate ➔ generate `outputs/submission.csv` ➔ verify against `solution_checker.xlsx`):

```bash
python run_pipeline.py --model gradient_boosting
```

Compare all supported classifiers on validation split:
```bash
python run_pipeline.py --compare-all
```

Supported `--model` options:
* `gradient_boosting` *(default, highest accuracy and AUC)*
* `decision_tree` *(fast, interpretable tree baseline)*
* `random_forest` *(ensemble bagging)*
* `logistic_regression` *(linear baseline)*

### Option 2: Interactive Exploration via Jupyter Notebook

Launch JupyterLab or Notebook to step through the interactive visualizations:
```bash
jupyter notebook notebooks/Bank_Term_Deposit_Analysis_and_Modeling.ipynb
```

---

## 💡 Strategic Business Recommendations for Bank Marketing Teams

1. **Prioritize Prior Responders (`poutcome == 'success'`):**
   Route clients with successful historical interactions directly to senior relationship managers. This group yields an exceptional **>60% conversion rate**.
2. **Segment Campaign Strategies by Occupation:**
   Tailor messaging for students and retirees. Retirees respond favorably to capital preservation and fixed-yield products, while younger savers favor digital setup.
3. **Dynamic Call Routing Based on Duration:**
   Telemarketers should be given conversion playbooks if a prospect remains on the phone past the **3-minute mark**, where engagement inflection occurs.
4. **Prune Cold Leads with Heavy Debt:**
   Filter out prospects carrying concurrent housing and personal loans with low account balances, saving call center bandwidth for higher-yield prospects.

---

## 📜 License & Acknowledgements

* **License:** Distributed under the [MIT License](LICENSE).
* **Author:** Vipin Kumar ([@vipinvicky127-svg](https://github.com/vipinvicky127-svg))
* **Dataset:** Based on the classic Bank Marketing Classification Benchmark.
