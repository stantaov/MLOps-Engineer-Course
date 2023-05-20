"""
Project: Predict Customer Churn with Clean Code

Description: The churn_library.py is a library of
functions to find customers who are likely to churn.

Author: Stan Taov

Date: May 20, 2023

Version: 0.0.2
"""

# import libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import Tuple, Any, Union, Optional
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import joblib
import toml
import os
import seaborn as sns
sns.set()


# Fix for QXcbConnection: Could not connect to display: Aborted
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# create logger
logger = logging.getLogger('churn_model')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.FileHandler('./logs/logging.log', 'w', 'utf-8')
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

config = toml.load('config.toml')

cat_columns = config['CATEGORY_COLUMNS']
plot_columns = config['PLOT_COLUMNS']


def import_data(pth: str) -> pd.DataFrame:
    """
    This function returns dataframe for the csv file found at pth

    Args:
        pth (str): The path to the csv file

    Returns:
        df (pd.DataFrame): Pandas dataframe

    Raises:
        FileNotFoundError: If the csv file is not found
        Exception: If unexpexted error occurred
    """

    try:
        logger.debug("Attempting to load %s file", pth)
        df = pd.read_csv(pth)
        logger.info("Loading the input data: SUCCESS")
        return df
    except FileNotFoundError as err:
        logger.error("Incorrect path %s, the input file is missing.", pth)
        raise err
    except Exception as err:
        logger.warning("Unexcpected error occurred: %s", err)
        raise err


def save_plots(column_names: list, df: pd.DataFrame) -> None:
    """
    This function saves EDA images to the target folder

    Args:
        column_names (list): a list of columns available in the dataframe,
        df (pd.DataFrame): target dataframge

    Returns:
        None, images are saved to the default location 'images/eda'

    Raises:
        AssertionError: When data types are not matching
        Exception: If unexpexted error occurred
    """
    try:
        logger.debug("Checking if df argument is pandas dataframe")
        assert isinstance(df, pd.DataFrame)
        logger.info("Argument df is pandas dataframe")
    except AssertionError as err:
        logger.error("Argument df is not pandas dataframe, but %s.", type(df))
        raise err
    except Exception as err:
        logger.warning("Unexcpected error occurred: %s", err)
        raise err
    try:
        logger.debug("Checking if column_names argument is a list type")
        assert isinstance(column_names, list)
        logger.info("Argument column_names is a list type")
    except AssertionError as err:
        logging.error(
            "Argument column_names is not a list, but %s.", type(column_names))
        raise err
    except Exception as err:
        logger.warning("Unexpected error occurred: %s", err)
        raise err

    for column_name in column_names:
        if column_name not in df.columns:
            logger.warning(
                "Column %s is not in dataframe, skipping.", column_name)
            continue

        plt.figure(figsize=(20, 10))

        if column_name == "Churn":
            logger.debug("Attempting to plot %s column", column_name)
            df[column_name].hist()
            logger.info("Plotting histogram for %s column", column_name)
        elif column_name == "Customer_Age":
            logger.debug("Attempting to plot %s column", column_name)
            df[column_name].hist()
            logger.info("Plotting histogram for %s column", column_name)
        elif column_name == "Marital_Status":
            logger.debug("Attempting to plot %s column", column_name)
            df[column_name].value_counts("normalize").plot(kind="bar")
            logger.info("Plotting bar chart for %s column", column_name)
        elif column_name == "Total_Trans_Ct":
            logger.debug("Attempting to plot %s column", column_name)
            sns.histplot(df[column_name], stat='density', kde=True)
            logger.info("Plotting histogram for %s column", column_name)
        elif column_name == "Heatmap":
            logger.debug("Attempting to plot %s column", column_name)
            sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
            logger.info("Plotting Heatmap for %s column", column_name)
        else:
            logger.warning(
                "Column %s doesn't match any plot condition, skipping.",
                column_name)
            continue

        plt.savefig(os.path.join("images", "eda", f"{column_name}.png"))
        plt.close()


def perform_eda(df: pd.DataFrame, plot_columns: list) -> pd.DataFrame:
    """
    This Function performs EDA on df and calls save_plots function
    to save figures to the image folder

    Args:
        df (pd.DataFrame): target dataframe
        plot_columns (list): list of columns for which plots will be created
    Returns:
        df (pd.DataFrame): dataframe with extra features

    Raises:
        AssertionError: When data types are not matching
        Exception: If unexpexted error occurred
    """

    try:
        logger.debug("Checking if df argument is pandas dataframe")
        assert isinstance(df, pd.DataFrame)
        logger.info("Argument df is pandas dataframe")
    except AssertionError as err:
        logger.error("Argument df is not pandas dataframe, but %s.", type(df))
        raise err
    except Exception as err:
        logger.warning("Unexpected error occurred: %s", err)
        raise err

    logger.info(
        "DataFrame has %s rows and %s columns",
        df.shape[0],
        df.shape[1])
    logger.info("Checking missing values %s.", df.isnull().sum())

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Generate and save EDA plots
    save_plots(plot_columns, df)

    return df


def encoder_helper(
        df: pd.DataFrame,
        category_lst: list,
        response: str) -> pd.DataFrame:
    """
    This helper function to turn each categorical column into a new column with
    propotion of churn for each category

    Args:
        df (pd.DataFrame): pandas dataframe
        category_lst (list): list of columns that contain categorical features
        response (str): string of response name
        [optional argument that could be used for naming variables or index y column]

    Returns:
        df (pd.DataFrame): pandas dataframe with new columns (Gender_Churn, Education_Level_Churn,
        Marital_Status_Churn, Income_Category_Churn, Card_Category_Churn)

    Raises:
        AssertionError: When data types are not matching
        Exception: If unexpexted error occurred
    """

    try:
        logger.debug("Checking if df argument is pandas dataframe")
        assert isinstance(df, pd.DataFrame)
        logger.info("Argument df is pandas dataframe")
    except AssertionError as err:
        logger.error(
            "Argument df is not a pandas dataframe, but %s.",
            type(df))
        raise err
    except Exception as err:
        logger.warning("Unexpected error occurred: %s", err)
        raise err

    try:
        logger.debug("Checking if category_lst argument is list")
        assert isinstance(category_lst, list)
        logger.info("Argument category_lst is list")
    except AssertionError as err:
        logger.error(
            "Argument column_names is not a list, but %s.", type(category_lst))
        raise err
    except Exception as err:
        logger.warning("Unexpected error occurred: %s", err)
        raise err

    try:
        logger.debug("Checking if response argument is str")
        assert isinstance(response, str)
        logger.info("Argument response is str")
    except AssertionError as err:
        logging.error(
            "Argument response is not a str, but %s.",
            type(response))
        raise err
    except Exception as err:
        logger.warning("Unexpected error occurred: %s", err)
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
                "New encoder %s column is created: SUCCESS.",
                category + '_' + response)

        except KeyError as err:
            logging.error("Incorrect column name")
            raise err

    return df


def perform_feature_engineering(df: pd.DataFrame,
                                response: str) -> Tuple[pd.DataFrame,
                                                        pd.DataFrame,
                                                        pd.Series,
                                                        pd.Series]:
    """
    This function performs feature engineering on the input DataFrame and split it into training and testing datasets.
    The function keeps only the relevant columns (specified in keep_cols) in the DataFrame,
    then splits the data into train and test sets for both X (features) and y (target).

    Args:
        df (pd.DataFrame): The input DataFrame to be transformed and split.
        response (str): The name of the response column in df.

    Returns:
        Tuple (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series): Returns four pandas data structures
        as a tuple in the following order:
        X_train: Features for the training set.
        X_test: Features for the testing set.
        y_train: Target for the training set.
        y_test: Target for the testing set.

    Raises:
        AssertionError: When data types are not matching
        Exception: If unexpexted error occurred
    """
    try:
        logger.debug("Checking if df argument is pandas dataframe")
        assert isinstance(df, pd.DataFrame)
        logger.info("Argument df is pandas dataframe")
    except AssertionError as err:
        logger.error(
            "Argument df is not a pandas dataframe, but %s.",
            type(df))
        raise err
    except Exception as err:
        logger.warning("Unexpected error occurred: %s", err)
        raise err

    try:
        logger.debug("Checking if response argument is pstr type")
        assert isinstance(response, str)
        logger.info("Argument response is str type")
    except AssertionError as err:
        logger.error("Argument response is not a str, but %s", type(df))
        raise err
    except Exception as err:
        logger.warning("Unexpected error occurred: %s", err)
        raise err

    y = df[response]

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

    X = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train: pd.DataFrame,
                                y_test: pd.DataFrame,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    This function produces a classification report for training and testing results and stores the report as an image
    in the images/results folder.

    Args:
        y_train (pd.Series): Training response values.
        y_test (pd.Series): Test response values.
        y_train_preds_lr (np.array): Training predictions from logistic regression.
        y_train_preds_rf (np.array): Training predictions from random forest.
        y_test_preds_lr (np.array): Test predictions from logistic regression.
        y_test_preds_rf (np.array): Test predictions from random forest.

    Returns:
        None

    Raises:
        AssertionError: When length of training data not matching
        Exception: If unexpexted error occurred
    """

    try:
        logger.debug(
            "Checking if y_train, y_train_preds_lr and y_train_preds_rf arguments have equal length")
        assert len(y_train) == len(y_train_preds_lr) == len(y_train_preds_rf)
        logger.info(
            "Argument y_train, y_train_preds_lr and y_train_preds_rf are equal")
    except AssertionError as err:
        logging.error("Number of data points for trainig data don't match")
        raise err
    except Exception as err:
        logger.warning("Unexpected error occurred: %s", err)
        raise err

    try:
        logger.debug(
            "Checking if y_test, y_test_preds_lr and y_test_preds_rf arguments have equal length")
        assert len(y_test) == len(y_test_preds_lr) == len(y_test_preds_rf)
        logger.info(
            "Argument y_test, y_test_preds_lr and y_test_preds_rf are equal")
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


def feature_importance_plot(
        model: Any,
        X_data: pd.DataFrame,
        output_pth: str) -> None:
    """
    This function creates a plot of feature importances and saves it as a .jpg file.

    Args:
        model (Any): A trained model object that contains `feature_importances_` attribute.
        X_data (pd.DataFrame): A pandas DataFrame of feature values.
        output_pth (str): A string representing the path where the feature importance plot image should be saved.

    Returns:
        None
    """

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


def plot_roc(
        model1: BaseEstimator,
        model2: BaseEstimator,
        X_test: Any,
        y_test: Any,
        output_pth: str) -> None:
    """
    This function plots ROC curves for two models and saves the result in an image file.

    Args:
        model1 (BaseEstimator): The first model for which to plot the ROC curve.
        model2 (BaseEstimator): The second model for which to plot the ROC curve.
        X_test (Any): Test features.
        y_test (Any): Test labels.
        output_pth (str): The path to store the image file.

    Returns:
        None
    """
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    _ = plot_roc_curve(model1, X_test, y_test, ax=ax, alpha=0.8)
    _ = plot_roc_curve(model2, X_test, y_test, ax=ax, alpha=0.8)
    plt.tight_layout()
    plt.savefig(f"images/{output_pth}/ROC_Curve_Results.jpg")
    plt.close()


def print_metrics(y_true: Union[np.ndarray,
                                pd.Series],
                  preds: Union[np.ndarray,
                               pd.Series],
                  model_name: Optional[str] = None) -> None:
    """
    This function prints the accuracy, precision, recall, and F1 score of the model.

    Args:
        y_true (Union[np.ndarray, pd.Series]): The actual y values in the dataset.
        preds (Union[np.ndarray, pd.Series]): The predictions for those values from a model.
        model_name (Optional[str], default=None): A name associated with the model for printing.

    Returns:
        None
    """
    if model_name:
        model_name = ' for ' + model_name
    else:
        model_name = ''

    print(f'Accuracy score{model_name}: {accuracy_score(y_true, preds):.2f}')
    print(f'Precision score{model_name}: {precision_score(y_true, preds):.2f}')
    print(f'Recall score{model_name}: {recall_score(y_true, preds):.2f}')
    print(f'F1 score{model_name}: {f1_score(y_true, preds):.2f}\n\n')


def train_models(X_train, X_test, y_train, y_test):
    '''
    This function trains, stores model results: images + scores, and saves models

    Args:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    Returns:
        None
    '''

    try:
        logger.debug(
            "Checking if training data has some non numerical records")
        assert X_train.shape[1] == X_train.select_dtypes(
            include=np.number).shape[1]
        logger.info("Training data contains only numerical data")
    except AssertionError as err:
        logger.error("Training data contains some non numerical records")
        raise err
    except Exception as err:
        logger.warning("Unexpected error occurred")
        raise err

    try:
        logger.debug("Checking if testing data has some non numerical records")
        assert X_test.shape[1] == X_test.select_dtypes(
            include=np.number).shape[1]
        logger.info("Testing data contains only numerical data")
    except AssertionError as err:
        logger.error("Testing data contains some non numerical records")
        raise err
    except Exception as err:
        logger.warning("Unexpected error occurred")
        raise err

    try:
        logger.debug(
            "Checking if training labels have some non numerical records")
        assert pd.DataFrame(y_train).shape[0] == pd.DataFrame(
            y_train).select_dtypes(include=np.number).shape[0]
        logger.info("Training labels contains only numerical data")
    except AssertionError as err:
        logger.error("Training labels contain some non numerical records")
        raise err
    except Exception as err:
        logger.warning("Unexpected error occurred")
        raise err

    try:
        logger.debug(
            "Checking if testing labels have some non numerical records")
        assert pd.DataFrame(y_test).shape[0] == pd.DataFrame(
            y_test).select_dtypes(include=np.number).shape[0]
        logger.info("Testing labels contains only numerical data")
    except AssertionError as err:
        logger.error("Testing labels contain some non numerical records")
        raise err
    except Exception as err:
        logger.warning("Unexpected error occurred")
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
    logger.info("Trained Random Forest model: SUCCESS")

    lrc.fit(X_train, y_train)
    logger.info("Trained Logistic Regression model: SUCCESS")

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
    plot_roc(lrc, cv_rfc.best_estimator_, X_test, y_test, "results")

    joblib.dump(cv_rfc.best_estimator_, "models/rfc_model.pkl")
    joblib.dump(lrc, "models/logistic_model.pkl")


if __name__ == "__main__":
    data = import_data("data/bank_data.csv")
    data_eda = perform_eda(data, plot_columns)
    encoded_df = encoder_helper(data_eda, cat_columns, "Churn")
    x_train_, x_test_, y_train_, y_test_ = perform_feature_engineering(
        encoded_df, "Churn")
    train_models(x_train_, x_test_, y_train_, y_test_)
