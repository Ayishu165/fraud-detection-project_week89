# 🛡️ Fraud Detection Project (10 Academy – Week 8 & 9)

An end-to-end machine learning pipeline to detect fraud from two real-world datasets.


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


