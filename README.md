# Heterogeneous Ensemble Model for Predicting Changes in Non-Investment Grade Short Term Debt

This repository presents an advanced ensemble modeling approach to predict percentage changes in non-investment grade short-term debt by using the ICE BofA US High Yield Index Option-Adjusted Spread as a proxy. The methodology combines Random Forest and Extra Trees classifiers to deliver robust and accurate predictions. Statistical analyses and visualizations are provided to enhance the interpretability of the results.

## Table of Contents
- [Overview](#Overview)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Selection](#feature-selection)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Results](#results)

## Overview
In this project, a heterogeneous ensemble model is employed to predict percentage changes in non-investment grade short-term debt. By combining Random Forest and Extra Trees classifiers, the ensemble leverages the strengths of both algorithms—Random Forest's robustness and Extra Trees' variance reduction. The predictions from these models are then integrated using logistic regression, which learns to weigh their contributions effectively. This approach improves predictive accuracy, reduces overfitting, and provides more reliable predictions, particularly when the model's confidence is high. 

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
   - Utilizes bootstrap aggregating to build multiple decision trees on different samples of the dataset.
   - Each tree is trained on a random subset of features, enhancing diversity among trees while reducing overfitting.
   - Final predictions are made by averaging the probabilities output by each tree or majority vote.

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

## Results

The results of training a model with the target variable as a -35 basis point change over the next 70 trading days are as follows:
- **Accuracy (Entire Test Set)**: 0.7779212395093609
- **AUC (Entire Test Set)**: 0.6683076749562963
- **Filtered Accuracy (Confidence > 0.5)**: 0.967032967032967


![image](https://github.com/evanwohl/creditAnalysis/assets/156111794/b401e468-b4f5-4de6-bc8f-b40a3d283558)
![image](https://github.com/evanwohl/creditAnalysis/assets/156111794/ba5ccc56-579b-4bb7-82bd-338c26805ee4)


The results of training a model with the target variable as a -28 basis point change over the next 70 trading days are as follows:
- **Accuracy (Entire Test Set)**: 0.7585539057456423
- **AUC (Entire Test Set)**: 0.6804820998125167
- **Filtered Accuracy (Confidence > 0.5)**: 0.9313304721030042

![image](https://github.com/evanwohl/creditAnalysis/assets/156111794/034de1de-f8ee-45a7-bc7a-ef4507b4dd2e)
![image](https://github.com/evanwohl/creditAnalysis/assets/156111794/1182e000-fcc9-4b91-a047-5c3bb298565b)


It is important to note the AUC can be increased to 0.7+ by parameter tuning, though other performance metrics may suffer.

## Summary 
The ensemble model demonstrates strong performance in predicting significant changes in non-investment grade short-term debt, particularly when the model's confidence is high. The time series analysis and scatter plot visualization both support the utility of confidence scores as an indicator of prediction reliability. While the overall accuracy and AUC scores are good, focusing on confidence scores provides a more actionable and reliable metric. The model's ability to track movements in debt changes effectively, even with a moderate AUC, highlights the importance of confidence filtering in practical applications.









   

