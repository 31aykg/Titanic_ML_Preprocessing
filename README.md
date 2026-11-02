# Titanic ML Preprocessing & EDA
Kaggle data surce - [text](https://www.kaggle.com/datasets/anandambastha/titanic/data)
## Project Overview

This project demonstrates fundamental data preprocessing and exploratory data analysis using Python.

## Dataset

Titanic dataset obtained from Kaggle.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Preprocessing Steps

1. Loaded the Titanic dataset
2. Inspected the dataset
3. Checked missing values
4. Handled missing values
5. Checked and removed duplicate records
6. Performed exploratory data analysis
7. Selected relevant features
8. Encoded categorical variables using One-Hot Encoding
9. Standardized numerical features using StandardScaler
10. Exported the cleaned dataset

## EDA

The analysis includes:

- Age distribution
- Fare distribution
- Survival distribution
- Survival by gender
- Survival by passenger class
- Correlation analysis

## Output

The cleaned dataset is saved as:

`cleaned_titanic_dataset.csv`

# Titanic Survival Prediction API

## Project Overview
An end-to-end Machine Learning application that predicts
Titanic passenger survival using Python and Flask.

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- Flask
- Joblib
- REST API
- Render

## Machine Learning Workflow
1. Load Titanic dataset.
2. Perform data preprocessing.
3. Train Logistic Regression and Random Forest.
4. Evaluate model performance.
5. Select the best model based on F1-score.
6. Serialize the complete pipeline using Joblib.
7. Build a Flask prediction API.
8. Test API endpoints.
9. Deploy the application.

## How to Run

Install dependencies:

pip install -r requirements.txt

Train model:

python train_model.py

Start API:

python app.py

## API Endpoints

GET /
GET /health
POST /predict

## Prediction Input

{
  "Pclass": 3,
  "Sex": "male",
  "Age": 25,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}

## Model Evaluation
The models are compared using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Deployment
(http://127.0.0.1:5000/)