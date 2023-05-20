"""
Project: Predict Customer Churn with Clean Code

Description: The churn_library.py is a library of
functions to find customers who are likely to churn.

Author: Stan Taov

Date: May 17, 2023

Version: 0.0.1
"""

# import libraries
import os
import logging
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Enable logging

# logging.basicConfig(
#         file = './logging.log',
#         level = logging.INFO,
#         filemode = 'w',
#         format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('./logs/logging.log', 'w', 'utf-8')
root_logger.addHandler(handler)

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth)
        logging.info("Loading the input data: SUCCESS")
        return df
    except FileNotFoundError as err:
        logging.error("Incorrect path %s, the input file is missing.", pth)
        raise err


def save_plots(column_names, df):
    '''
    input:
            column_names: a list of columns available in the dataframe,
            df: target dataframge
    output:
            saves EDA images in the target folder
    '''
    try:
        assert isinstance(df, pd.DataFrame)
    except AssertionError as err:
        logging.error("Argument df is not pandas dataframe, but %s.", type(df))
        raise err
    try:
        assert isinstance(column_names, list)
    except AssertionError as err:
        logging.error(
            "Argument column_names is not a list, but %s.", type(column_names))
        raise err

    for column_name in column_names:
        if column_name not in df.columns:
            logging.info(
                "Column %s is not in dataframe, skipping.", column_name)
            continue

        plt.figure(figsize=(20, 10))

        if column_name == "Churn":
            df[column_name].hist()
        elif column_name == "Customer_Age":
            df[column_name].hist()
        elif column_name == "Marital_Status":
            df[column_name].value_counts("normalize").plot(kind="bar")
        elif column_name == "Total_Trans_Ct":
            sns.histplot(df[column_name], stat='density', kde=True)
        elif column_name == "Heatmap":
            sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
        else:
            logging.info(
                "Column %s doesn't match any plot condition, skipping.", column_name)
            continue

        plt.savefig(os.path.join("images", "eda", f"{column_name}.png"))
        plt.close()


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    try:
        assert isinstance(df, pd.DataFrame)
    except AssertionError as err:
        logging.error("Argument df is not pandas dataframe, but %s.", type(df))
        raise err

    logging.info("DataFrame has %s rows and %s columns", df.shape[0], df.shape[1])
    logging.info("Checking missing values %s.", df.isnull().sum())

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    # Generate and save EDA plots
    column_names = [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        "Total_Trans_Ct",
        "Heatmap"]
    save_plots(column_names, df)

    return df


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for

    '''
    try:
        assert isinstance(df, pd.DataFrame)
    except AssertionError as err:
        logging.error("Argument df is not a pandas dataframe, but %s.", type(df))
        raise err
    try:
        assert isinstance(category_lst, list)
    except AssertionError as err:
        logging.error(
            "Argument column_names is not a list, but %s.", type(category_lst))
        raise err
    try:
        assert isinstance(response, str)
    except AssertionError as err:
        logging.error("Argument response is not a str, but %s.", type(response))
        raise err

    for category in category_lst:
        try:
            assert category in [
                'Gender',
                'Education_Level',
                'Marital_Status',
                'Income_Category',
                'Card_Category']
            category_new_lst = []
            category_groups = df.groupby(category).mean()['Churn']
            for i in df[category]:
                category_new_lst.append(category_groups.loc[i])
            df[f"{category}_{response}"] = category_new_lst
            logging.info(
                "New encoder %s column is created: SUCCESS.", category+'_'+response)

        except KeyError as err:
            logging.error("Incorrect column")
            raise err
        
    return df 



def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name
              [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    try:
        assert isinstance(df, pd.DataFrame)
    except AssertionError as err:
        logging.error("Argument df is not a pandas dataframe, but %s.", type(df))
        raise err
    try:
        assert isinstance(response, str)
    except AssertionError as err:
        logging.error("Argument response is not a str, but %s", type(df))
        raise err

    y = df[response]
    X = pd.DataFrame()

    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn"]

    X[keep_cols] = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    try:
        assert len(y_train) == len(y_train_preds_lr) == len(y_train_preds_rf)
    except AssertionError as err:
        logging.error("Number of data points for trainig data don't match")
        raise err
    try:
        assert len(y_test) == len(y_test_preds_lr) == len(y_test_preds_rf)
    except AssertionError as err:
        logging.error("Number of data points for testing data don't match")
        raise err

    classification_reports_data = {
        "Random_Forest": [
            [
                "Random Forest Train", classification_report(
                    y_train, y_train_preds_rf)], [
                "Random Forest Test", classification_report(
                    y_test, y_test_preds_rf)]], "Logistic_Regression": [
            [
                "Logistic Regression Train", classification_report(
                    y_train, y_train_preds_lr)], [
                "Logistic Regression Test", classification_report(
                    y_test, y_test_preds_lr)]]}

    for model, classification_data in classification_reports_data.items():
        plt.rc("figure", figsize=(10, 5))
        plt.text(0.01, 1, str(classification_data[0][0]), {
                 "fontsize": 10}, fontproperties="monospace")
        plt.text(0.01, 0.6, str(classification_data[0][1]), {
                 "fontsize": 10}, fontproperties="monospace")
        plt.text(0.01, 0.5, str(classification_data[1][0]), {
                 "fontsize": 10}, fontproperties="monospace")
        plt.text(0.01, 0.1, str(classification_data[1][1]), {
                 "fontsize": 10}, fontproperties="monospace")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"images/results/{model}.jpg")
        plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(f"images/{output_pth}/Feature_Importance.jpg")
    plt.close()


def print_metrics(y_true, preds, model_name):
    '''
    input:
    y_true - the y values that are actually true in the dataset (NumPy array or pandas series)
    preds - the predictions for those values from some model (NumPy array or pandas series)
    model_name - (str - optional) a name associated with the model if you would like to print it

    output:
    None - prints the accuracy, precision, recall, and F1 score
    '''

    print(
        'Accuracy score for ' +
        model_name +
        ' :',
        format(
            accuracy_score(
                y_true,
                preds)))
    print(
        'Precision score ' +
        model_name +
        ' :',
        format(
            precision_score(
                y_true,
                preds)))
    print(
        'Recall score ' +
        model_name +
        ' :',
        format(
            recall_score(
                y_true,
                preds)))
    print('F1 score ' + model_name + ' :', format(f1_score(y_true, preds)))
    print('\n\n')


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    try:
        assert X_train.shape[1] == X_train.select_dtypes(
            include=np.number).shape[1]
    except AssertionError as err:
        logging.error("Training data contains some non numerical records")
        raise err
    try:
        assert X_test.shape[1] == X_test.select_dtypes(
            include=np.number).shape[1]
    except AssertionError as err:
        logging.error("Testing data contains some non numerical records")
        raise err
    try:
        assert pd.DataFrame(y_train).shape[0] == pd.DataFrame(
            y_train).select_dtypes(include=np.number).shape[0]
    except AssertionError as err:
        logging.error("Training labels contain some non numerical records")
        raise err
    try:
        assert pd.DataFrame(y_test).shape[0] == pd.DataFrame(
            y_test).select_dtypes(include=np.number).shape[0]
    except AssertionError as err:
        logging.error("Testing labels contain some non numerical records")
        raise err

    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=1000)

    # Defining Hyper Parameters
    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"]
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    logging.info("Trained Random Forest model: SUCCESS")

    lrc.fit(X_train, y_train)
    logging.info("Trained Logistic Regression model: SUCCESS")

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    print("Train random forest model")
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    # Print Random Forest scores
    print(print_metrics(y_test, y_test_preds_rf, "Random Forest"))

    y_train_preds_lr = lrc.predict(X_train)
    print("Train logistic regression model")
    y_test_preds_lr = lrc.predict(X_test)
    # Print Logistic Regression scores
    print(print_metrics(y_test, y_test_preds_lr, "Logistic Regression"))

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(cv_rfc, X_test, "results")

    joblib.dump(cv_rfc.best_estimator_, "models/rfc_model.pkl")
    joblib.dump(lrc, "models/logistic_model.pkl")


if __name__ == "__main__":
    data = import_data("data/bank_data.csv")
    data_eda = perform_eda(data)
    encoded_df = encoder_helper(data_eda, cat_columns, "Churn")
    x_train_, x_test_, y_train_, y_test_ = perform_feature_engineering(
        encoded_df, "Churn")
    train_models(x_train_, x_test_, y_train_, y_test_)
