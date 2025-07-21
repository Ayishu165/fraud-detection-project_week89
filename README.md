# 🛡️ Fraud Detection Project (10 Academy – Week 8 & 9)

An end-to-end machine learning pipeline to detect fraud from two real-world datasets.
# 🧠  Task 1 EDA

1. Load Raw Datasets
Fraud_Data.csv: Transaction and user metadata (IP, device, age, etc.)

IpAddress_to_Country.csv: IP ranges with country mapping

creditcard.csv: Financial transaction dataset

 2. Handle Missing Values
Checked for missing values in all datasets.

No missing values found, so no imputation or dropping was needed.

3. Data Cleaning
Removed duplicate rows.


 4. Exploratory Data Analysis (EDA)
Performed basic analysis using visualizations:

Univariate Analysis: Histograms for age, purchase_value, etc.

Bivariate Analysis: Fraud vs source, browser, age groups.


5. Merge Datasets for Geolocation Analysis
Converted IP addresses to integer format.

Merged Fraud_Data with IpAddress_to_Country based on IP range match.

Added a new column country to the fraud dataset.


6. Feature Engineering
Added several derived features:

hour_of_day and day_of_week from purchase_time

time_since_signup = difference between purchase_time and signup_time

transaction_count_per_user and avg_time_between_txn




7. Data Transformation
Class Imbalance: Applied SMOTE to oversample minority fraud cases.

Feature Scaling: Used StandardScaler to normalize numeric features.

Categorical Encoding: Applied one-hot encoding to browser, source, sex, and country.



# 🧠 Task 2 – Model Building and Training (Fraud Detection Project)

This module focuses on training and evaluating machine learning models for fraud detection using two datasets: a credit card fraud dataset and an online transaction dataset.

---

  Objectives
The primary goals accomplished in this phase were:

Data Preparation: Separate features and target variables, and perform stratified train-test splits on both datasets.

Model Selection: Build and compare two distinct classification models:

Logistic Regression: As a simple, interpretable baseline.

Random Forest Classifier: As a powerful ensemble model.

Model Training and Evaluation: Train both models on two different fraud datasets and evaluate their performance using metrics specifically suited for imbalanced data (Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC, PR-AUC).

Model Justification: Clearly identify and justify the "best" performing model based on the evaluation results.

Step 3: Data Sources
Two distinct datasets were utilized for this task:

Credit Card Fraud Data: Loaded from ../data/cleaned_credit_data.csv. This dataset represents credit card transactions, with a very small percentage of fraudulent cases.

Online Fraud Data: Loaded from pre-processed files: ../data/X_train_scaled.csv, ../data/X_test_scaled.csv, ../data/y_train_smote.csv, and ../data/y_test_original.csv. This dataset represents online transactions and has undergone scaling and SMOTE (Synthetic Minority Over-sampling Technique) on the training labels to address class imbalance.

Step 4: Data Preparation Steps
For both datasets, the following data preparation steps were executed:

Feature and Target Separation:

For the Credit Card data, the 'Class' column was identified as the target variable (y), and all other columns formed the feature set (X).

For the Online Fraud data, X_train, X_test, y_train, and y_test were loaded directly as pre-separated and pre-processed files.

Train-Test Split:

For the Credit Card data, sklearn.model_selection.train_test_split was used with test_size=0.3 (30% for testing) and random_state=42 for reproducibility.

Stratification (stratify=y): This crucial parameter was used to ensure that the proportion of fraudulent transactions in both the training and testing sets remained consistent with the original dataset's distribution. This is vital for reliable evaluation of models on imbalanced data.

SMOTE Application (Online Fraud Data): The y_train_smote.csv file indicates that the training labels for the online fraud dataset were artificially balanced using SMOTE, which generates synthetic samples of the minority class to improve model learning.

Step 5: Model Selection and Implementation
Two classification models were selected and implemented using the scikit-learn library:

Logistic Regression:

Implementation: sklearn.linear_model.LogisticRegression(max_iter=1000, random_state=42)

Rationale: Chosen as a simple, linear, and highly interpretable baseline model to provide a fundamental benchmark.

Random Forest Classifier:

Implementation: sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

Rationale: Selected as a powerful ensemble method known for its high accuracy, robustness to overfitting, and effectiveness in handling complex and imbalanced datasets.

Step 6: Model Training and Evaluation
Both models were trained on their respective training sets (X_train, y_train) using the .fit() method. Model performance was then evaluated on the unseen test sets (X_test, y_test) using a comprehensive set of metrics tailored for imbalanced data:

Confusion Matrix: Provides a detailed breakdown of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN). This is essential for understanding the types of errors (missed frauds vs. false alarms).

Classification Report: Presents Precision, Recall (Sensitivity), and F1-Score for each class.

Precision: The accuracy of positive predictions (how many flagged as fraud were actually fraud).

Recall: The ability to find all positive samples (how many actual frauds were caught).

F1-Score: The harmonic mean of precision and recall, balancing both.

ROC-AUC Score: Measures the classifier's ability to distinguish between classes.

PR-AUC Score (Average Precision Score): This is a critical metric for imbalanced datasets. It focuses on the minority class and measures the trade-off between precision and recall across different thresholds. A higher PR-AUC indicates better performance in identifying the positive class without generating an excessive number of false positives.


# 🧠  Task 3 Model Explainability

This README summarizes Task 3 of the Fraud Detection Project, focusing on understanding why our best model makes its predictions.

 This task aims to demystify our best-performing fraud detection model by explaining its decision-making process.

To interpret the Random Forest Classifier, identified as the top model in Task 2, using advanced explainability techniques.

Best Model: The optimally tuned Random Forest Classifier from the online fraud dataset is the focus of this explainability analysis.

 We utilize SHAP (SHapley Additive exPlanations), a powerful game-theoretic approach for model interpretability.

SHAP Values: SHAP values quantify how much each feature contributes to a prediction, pushing it higher or lower than the average prediction.

Global Insights (SHAP Summary Plot): This plot reveals the most important features across the entire dataset, showing their overall impact and direction (e.g., high transaction amount increases fraud likelihood).

Local Insights (SHAP Force Plot): This plot explains individual predictions, visualizing exactly which features pushed a specific transaction towards being classified as fraudulent or legitimate.

Key Drivers of Fraud (Expected): Based on typical fraud patterns, SHAP plots are expected to highlight transaction Amount, Time anomalies, and potentially specific derived features (e.g., V17, V14) as primary indicators of fraud.

Value of Explainability: These insights build trust in the model, provide actionable intelligence for fraud prevention strategies, and guide future model improvements.

How to Run: SHAP analysis would be performed on the best-trained Random Forest model (from hyperparameter_tuning_3.ipynb) using the X_test_scaled.csv and y_test_original.csv data.



