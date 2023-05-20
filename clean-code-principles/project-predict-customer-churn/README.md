# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The objective of this project is to detect credit card customers who are at high risk of churning. The churn_notebook.ipynb has been cleaned up to ensure that the final project adheres to coding (PEP8) and engineering best practices. The end result will be a Python package that follows these standards.

## Files and data description

-   data
    -   bank_data.csv
-   images
    -   eda
        -   churn_distribution.png
        -   customer_age_distribution.png
        -   heatmap.png
        -   marital_status_distribution.png
        -   total_transaction_distribution.png
    -   results
        -   feature_importance.png
        -   logistics_results.png
        -   rf_results.png
        -   roc_curve_result.png
-   logs
    -   churn_library.log
-   models
    -   logistic_model.pkl
    -   rfc_model.pkl
-   churn_library.py
-   churn_notebook.ipynb
-   churn_script_logging_and_tests.py
-   README.md

### Important Files

**churn_library.py:**

This is a library of functions designed to identify customers who are at high risk of churning.

**churn_script_logging_and_tests.py:**  

### Data Description

The churn_library.py functions are accompanied by unit tests to ensure their accuracy and reliability. Additionally, any errors and INFO messages will be logged for future reference.

Source of dataset: [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)

The dataset was obtained from the website  [https://leaps.analyttica.com/home](https://leaps.analyttica.com/home)  and contains information on 10,000 credit card customers. The dataset includes various features such as age, salary, marital status, credit card limit, credit card category, and more. There are a total of 18 features in the dataset.

The objective of this dataset is to predict which customers are likely to churn. The dataset contains only 16.07% of customers who have churned, making it challenging to train a model to accurately predict churn. The dataset was obtained to help a bank manager who is concerned about the increasing number of customers leaving their credit card services. The manager hopes to use the predictions to proactively reach out to customers and provide better services to prevent churn.


## Project Diagram 

![Project Diagram](https://github.com/stantaov/MLOps-Engineer-Course/blob/main/clean-code-principles/project-predict-customer-churn/project_diagram.jpeg)

The code in churn_library.py should complete data science solution process including:

-   EDA
-   Feature Engineering (including encoding of categorical variables)
-   Model Training
-   Prediction
-   Model Evaluation

## Running Files

1.  Install Conda:

    `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
    `chmod +x Miniconda3-latest-Linux-x86_64.sh`
    `./Miniconda3-latest-Linux-x86_64.sh`

2.  Add Conda Forge Repo:

    `conda config --append channels conda-forge`

3.  Create Environment:

    `conda create --name churn_predict python=3.8`
    `conda activate churn_predict`

4.  Install Packages:

    `conda install --file requirements.txt`

5.  Run Churn Prediction:

    `python churn_library.py`

6.  Test Churn Prediction:

    `pytest churn_script_logging_and_tests.py`
