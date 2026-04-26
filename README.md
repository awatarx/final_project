# Heart Disease Classification Project

**Author:** Awatar Gautam  
**Repository:** [https://github.com/awatarx/final_project](https://github.com/awatarx/final_project)  
**Timeline:** Spring 2026

## Overview
This repository contains the final project for evaluating and predicting heart disease presence using the Heart Disease UCI dataset. The goal is to build a supervised machine learning model that accurately predicts the target variable based on patient health metrics.

## Methodology
1. **Data Preprocessing & EDA:** Cleaned the 303 samples, encoded categorical features, and applied standard scaling.
2. **Modeling:** Implemented and tuned Logistic Regression and Random Forest classifiers.
3. **Evaluation:** Models evaluated using Accuracy, Precision, Recall, and the F1-score to account for class representation.

## Project Structure
```
final_project/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── data/
│   ├── heart.csv                     # Original dataset
│   └── heart_cleaned.csv             # Cleaned dataset
├── notebooks/
│   ├── 01_data_prep_eda.ipynb        # Data preprocessing and EDA
│   └── 02_model_training.ipynb       # Model training notebook
├── src/
│   ├── data_prep.py                  # Data preprocessing script
│   ├── train_evaluate.py             # Model training and evaluation script
│   └── paper/
│       └── images/                   # Paper figures and visualizations
├── paper/
│   └── images/                       # Paper images
└── env/                              # Virtual environment (created after setup)
```

## How to Set Up and Run

### Step 1: Create a Virtual Environment
A virtual environment keeps project dependencies isolated from your system Python.

**On Windows:**
```bash
python -m venv env
```

**On macOS/Linux:**
```bash
python3 -m venv env
```

### Step 2: Activate the Virtual Environment

**On Windows:**
```bash
env\Scripts\activate
```

**On macOS/Linux:**
```bash
source env/bin/activate
```

When activated, you should see `(env)` at the beginning of your terminal prompt.

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Data Preprocessing
```bash
python src/data_prep.py
```
This script cleans the raw heart disease dataset and prepares it for modeling.

### Step 5: Run Model Training and Evaluation
```bash
python src/train_evaluate.py
```
This script trains the Logistic Regression and Random Forest classifiers and outputs performance metrics.

### Step 6 (Optional): Explore Jupyter Notebooks
If you want to interactively explore the analysis:
```bash
jupyter notebook notebooks/
```
Then open:
- `01_data_prep_eda.ipynb` - Data preprocessing and exploratory data analysis
- `02_model_training.ipynb` - Model training and evaluation

### Deactivate Virtual Environment
When finished, deactivate the virtual environment:
```bash
deactivate
```