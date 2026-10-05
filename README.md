# Task 3 - Linear Regression

## Objective

The objective of this task is to implement and understand simple and multiple linear regression.

## Dataset

Boston Housing Dataset.

## Tools Used

- Python
- Pandas
- Matplotlib
- Scikit-learn

## Data Preparation

The dataset was first explored to check its structure, shape, summary statistics, and missing values.

The features were separated into:

- **X** → input features used by the model
- **y** → target variable (`medv`), representing house price

The data was then divided into:

- **80% training data**
- **20% testing data**


## Linear Regression Models

### Simple Linear Regression

A simple linear regression model was created using only the `rm` feature to predict `medv`.

### Multiple Linear Regression

A multiple linear regression model was created using all available features to predict `medv`.

Both models were trained using the training dataset and evaluated on the testing dataset.

## Model Evaluation

The models were evaluated using three metrics:

- **MAE (Mean Absolute Error)** — measures the average absolute prediction error.
- **MSE (Mean Squared Error)** — squares the errors before averaging, giving larger errors more weight.
- **R² Score** — measures how well the model explains the variation in the target variable.

A regression line was also plotted for the simple linear regression model.



## Project Structure

```text
Task-3-Linear-Regression
│
├── data
│   └── housing.csv
│
├── task3_linear_regression.py
├── README.md
└── .venv