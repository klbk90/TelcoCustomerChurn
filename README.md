Project Overview
This project aims to predict customer churn for a telecommunications company using machine learning models. Customer churn is a critical metric for telecom companies as retaining existing customers is more cost-effective than acquiring new ones. By predicting which customers are likely to churn, the company can take proactive measures to improve customer retention.

Dataset
The dataset used for this project is the Telco Customer Churn dataset, which contains information about a telecom company's customers and their service usage. The dataset includes various features such as customer demographics, account information, and usage details, along with a binary target variable indicating whether the customer has churned.

Features
customerID: Unique identifier for each customer.
gender: Gender of the customer (Male/Female).
SeniorCitizen: Indicates if the customer is a senior citizen (1, 0).
Partner: Indicates if the customer has a partner (Yes/No).
Dependents: Indicates if the customer has dependents (Yes/No).
tenure: Number of months the customer has been with the company.
PhoneService: Indicates if the customer has phone service (Yes/No).
MultipleLines: Indicates if the customer has multiple lines (Yes/No/No phone service).
InternetService: Type of internet service (DSL, Fiber optic, No).
OnlineSecurity: Indicates if the customer has online security (Yes/No/No internet service).
OnlineBackup: Indicates if the customer has online backup (Yes/No/No internet service).
DeviceProtection: Indicates if the customer has device protection (Yes/No/No internet service).
TechSupport: Indicates if the customer has tech support (Yes/No/No internet service).
StreamingTV: Indicates if the customer has streaming TV (Yes/No/No internet service).
StreamingMovies: Indicates if the customer has streaming movies (Yes/No/No internet service).
Contract: Type of contract (Month-to-month, One year, Two year).
PaperlessBilling: Indicates if the customer has paperless billing (Yes/No).
PaymentMethod: Payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)).
MonthlyCharges: Monthly charges of the customer.
TotalCharges: Total charges incurred by the customer.
Churn: Target variable indicating if the customer has churned (Yes/No).
Project Steps
1. Data Preprocessing
Loading Data: Load the Telco Customer Churn dataset.
Exploratory Data Analysis (EDA): Explore the dataset to understand the distribution of features and the target variable.
Data Cleaning: Handle missing values and correct data types.
Feature Engineering: Create new features and encode categorical variables.
2. Handling Imbalanced Data
Resampling: Use SMOTEENN (Synthetic Minority Over-sampling Technique and Edited Nearest Neighbors) to handle imbalanced classes.
3. Model Building
Train-Test Split: Split the data into training and testing sets.
Model Selection: Evaluate multiple machine learning models to find the best performing model. Models include:
Random Forest
Gradient Boosting
Support Vector Machine (SVM)
Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
AdaBoost
XGBoost
Naive Bayes
4. Hyperparameter Tuning
GridSearchCV and RandomizedSearchCV: Perform hyperparameter tuning to optimize model performance.
5. Model Evaluation
Accuracy Score: Evaluate model performance using accuracy as the metric.
Best Model Selection: Identify the model with the highest accuracy on the test set.
Results
Best Model: The model with the highest accuracy.
Accuracy: The accuracy of the best model on the test set.
Usage
To reproduce the results of this project, follow these steps:

Clone the repository:

sh
Копировать код
git clone https://github.com/klbk90/TelcoCustomerChurn.git
cd TelcoCustomerChurn
Install the required dependencies:

sh
Копировать код
pip install -r requirements.txt
Run the project:

sh
Копировать код
python churn_prediction.py
Repository Structure
churn_prediction.py: Main script to run the churn prediction pipeline.
requirements.txt: List of required Python packages.
README.md: Project overview and instructions.
Conclusion
This project demonstrates the application of various machine learning models to predict customer churn in the telecommunications industry. By identifying customers at risk of churning, the company can take proactive steps to improve customer retention and reduce churn rates.
