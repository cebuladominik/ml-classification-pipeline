# End-to-End Machine Learning Pipeline for High-Dimensional Data Classification

## Project Overview
This repository contains a complete, end-to-end Machine Learning pipeline built in Python. The objective of the project was to develop a highly accurate classification system capable of processing large, high-dimensional biological datasets, extracting key features, and predicting outcomes with minimal error. 

The project demonstrates practical skills in **data preprocessing, dimensionality reduction, handling highly imbalanced datasets, and predictive modeling**

## Tech Stack & Tools
**Language**: Python
**Data Manipulation & Analysis:** Pandas, NumPy
**Machine Learning:** Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression, SVM, k-NN)
**Data Visualization:** Matplotlib, Seaborn
**Statistical Methods:** Variance Stabilizing Transformation (VST), Mutual Information algorithm

## Key Methodologies Implemented

### 1. Handling Imbalanced Classes
The raw dataset contained 609 records with a severe class imbalance (approximately 88% to 12% ratio). To prevent the model from favoring the majority class, dynamic class weighting (`class_weight='balanced'`) was implemented across all algorithms.

### 2. Dimensionality Reduction & Feature Selection
The initial dataset featured tens of thousands of variables (60,000+ features). To eliminate statistical noise and prevent overfitting, I applied the Mutual Information algorithm to extract the top 100 most critical features with the highest predictive power.

### 3. Data Transformation
Applied Variance Stabilizing Transformation (VST) to correct heteroscedasticity and stabilize variance across the dataset, preparing the data for optimal performance with linear models.

## Predictive Modeling
Five different supervised machine learning algorithms were trained and evaluated on an isolated test set (20% of the data) using stratified splitting to ensure data integrity. 

## Results

All five of supervised ML algorithms predicted outcomes with minimal error, each one scoring over 98% accuracy.

## Visualizations
The project includes robust data visualization to communicate insights effectively, such as Volcano Plots (for visualizing the statistical significance and magnitude of feature changes), confusion matrics (for providing a transparent breakdown of model predictions) and Box plots (for representing the distribution of features after Variance Stabilizing).
