# creditAnalysis

# Heterogeneous Ensemble Model for Predicting Changes in Non-Investment Grade Short Term Debt

This repository presents an advanced ensemble modeling approach to predict percentage changes in non-investment grade short-term debt by using the ICE BofA US High Yield Index Option-Adjusted Spread as a proxy. The methodology combines Random Forest and Extra Trees classifiers to deliver robust and accurate predictions. Comprehensive statistical analyses and visualizations are provided to enhance the interpretability of the results.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Selection](#feature-selection)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)

## Introduction

This project aims to develop an ensemble model to predict significant percentage changes in non-investment grade short-term debt. By leveraging the strengths of Random Forest and Extra Trees classifiers, the model provides accurate predictions and insights into the underlying factors influencing debt changes.

## Installation

To set up this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/evanwohl/creditAnalysis.git
cd creditAnalysis
pip install -r requirements.txt
```
## Usage
1. **Load Data**: Execute ```python3 dataloader.py --api_key=YOUR_FRED_API_KEY``` to download the required data 
2. **Run the Script**: Execute ```python3 ensemble.py``` to load data, train the model, and generate prediction 
3. **View Results**: Access and analyze the results that have been output in the terminal and as plots.

## Methodology

### Data Preprocessing

The dataset undergoes several preprocessing aand feature engineering steps to prepare it for modeling:

1. **Date Parsing**: Convert date columns to datetime objects.
2. **Lag Features**: Create lagged features for each time series.
3. **Volatility Calculation**: Calculate rolling standard deviations to capture volatility.
4. **Binary Change Calculation**: Generate a binary target variable indicating significant changes in debt.

### Feature Selection

Extra Trees Classifier is used to select important features, reducing dimensionality and improving model performance.

### Model Training

The modeling pipeline consists of:

1. **Random Forest Classifier**:
   - An ensemble method leveraging multiple decision trees.
   - Utilizes bootstrap aggregatingto build multiple decision trees on different samples of the dataset.
   - Each tree is trained on a random subset of features, enhancing diversity among the trees and reducing overfitting.
   - Final predictions are made by averaging the probabilities output by each tree (for classification) or taking the majority vote.

2. **Extra Trees Classifier**:
   - An ensemble technique similar to Random Forest but with increased randomness.
   - Splits nodes by choosing the best split among a random subset of features.
   - The added randomness typically reduces variancewhen compared to Random Forest.

3. **Logistic Regression Ensemble**:
   - Combines predictions from both Random Forest and Extra Trees classifiers.
   - Uses the predicted probabilities from both classifiers as features to train a logistic regression model.
   - The logistic regression model learns to weigh the contributions of each classifier's predictions to improve overall performance.

### Model Evaluation

The model is evaluated using several metrics:

- **Accuracy**: Measures the proportion of correct predictions.
- **AUC**: Assesses the model's ability to distinguish between classes.
- **Time Series Analysis**: Plots confidence scores and percentage changes over time.
- **Confidence vs Percentage Change**: Scatter plots illustrating the relationship between confidence scores and percentage changes.

   

