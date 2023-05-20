"""
Project: Predict Customer Churn with Clean Code

Description: churn_script_logging_and_tests.py contains
unit tests for the churn_library.py functions

Author: Stan Taov

Date: May 17, 2022

Version: 0.0.1
"""
import unittest
import logging
import glob
import joblib
import pandas as pd
import toml
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering, train_models

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


config = toml.load("config.toml")

path = config['PATH']
category_columns = config['CATEGORY_COLUMNS']
eda_image_paths = config['EDA_IMAGE_PTHS']
model_image_paths = config['MODEL_IMAGE_PTHS']

class TestChurn(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.df = import_data(path)
            self.df['Churn'] = self.df['Attrition_Flag'].apply(
                lambda val: 0 if val == "Existing Customer" else 1)
            logging.info("Testing import_data: SUCCESS")
        except FileNotFoundError as err:
            logging.error("ERROR: The input file is missing")
            raise err


    def test_import(self):
        '''
        test data import - this example is completed for you to assist with the other test functions
        '''
        logging.info("Testing test_import function")

        try:
            assert self.df.shape[0] > 0
            assert self.df.shape[1] > 0
        except AssertionError as err:
            logging.error("ERROR: Dataframe is missing some data")
            raise err


    def test_eda(self):
        '''
        test perform eda function
        '''
        logging.info("Testing perform_eda function")

        perform_eda(self.df)

        image_pths = glob.glob('images/eda/*.png')

        try:
            for pth in eda_image_paths:
                assert pth in image_pths
                logging.info("All required EDA images were created: SUCCESS")
        except AssertionError as err:
            logging.error(
                'ERROR: Required EDA image %s was not created by perform_eda() function',
                pth)
            raise err


    def test_encoder_helper(self):
        '''
        test encoder helper
        '''
        logging.info("Testing encoder_helper function")

        try:
            print(self.df)
            self.encoded_df = encoder_helper(self.df, category_columns, "Churn")
        except AssertionError as err:
            logging.error(
                "ERROR: Some function arguments were not provided")
            raise err

        for col in category_columns:
            try:
                assert f"{col}_Churn" in self.encoded_df.columns
                logging.info("%s column was created: SUCCESS", col)
            except AssertionError as err:
                logging.error("ERROR: %s column is missing", col)
                raise err


    def test_perform_feature_engineering(self):
        '''
        test perform_feature_engineering
        '''
        logging.info("Testing perform_feature_engineering")
        
        self.encoded_df = encoder_helper(
            df=self.df,
            category_lst=category_columns,
            response='Churn')

        try:
            X_train, X_test, y_train, y_test = perform_feature_engineering(
                self.encoded_df, response='Churn')
        except TypeError as err:
            logging.error(
                "ERROR: Incorrect input dataframe was proviced")
            raise err

        try:
            assert len(X_train) == len(y_train)
            assert len(X_test) == len(y_test)
            assert isinstance(X_train, pd.DataFrame)
            assert isinstance(X_test, pd.DataFrame)
            assert isinstance(y_train, pd.Series)
            assert isinstance(y_test, pd.Series)
            logging.info("Train/Test data were created: SUCCESS")
        except AssertionError as err:
            logging.error("ERROR: Train/Test data has wrong data format")
            raise err


    def test_train_models(self):
        '''
        test perform_feature_engineering
        '''
        logging.info("Testing train_models function")
        
        self.encoded_df = encoder_helper(
            df=self.df,
            category_lst=category_columns,
            response="Churn")

        X_train, X_test, y_train, y_test = perform_feature_engineering(self.encoded_df, "Churn")

        try:
            train_models(X_train, X_test, y_train, y_test)
        except NameError as err:
            logging.error(
                "ERROR: You haven't defined one of the required args in train_models()")
            raise err

        image_pths = glob.glob('images/results/*.jpg')
        print(image_pths)
        try:
            for pth in model_image_paths:
                assert pth in image_pths
        except AssertionError as err:
            logging.error(
                'ERROR: Something went wrong, not all model images were generated, %s',
                pth)
            raise err

        try:
            joblib.load('models/rfc_model.pkl')
            joblib.load('models/logistic_model.pkl')
            logging.info("Testing testing_models: SUCCESS")
        except FileNotFoundError as err:
            logging.error("ERROR: Some models were not generated")
            raise err


if __name__ == "__main__":
    unittest.main()
