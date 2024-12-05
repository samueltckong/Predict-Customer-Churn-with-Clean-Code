"""
Test suite for churn_library_solution functions.
"""
import os
import logging
import churn_library as cl

# Configure logging
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_import():
    """
    Test the data import function from churn_library.
    
    Verifies that the returned dataframe is not empty.

    Args:
        None

    Returns:
        None
    """
    try:
        df = cl.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
        assert df.shape[0] > 0 and df.shape[1] > 0
        logging.info("Testing import_data: Dataframe loaded with rows and columns")
    except FileNotFoundError as err:
        logging.error("Testing import_data: File not found")
        raise err
    except AssertionError as err:
        logging.error("Testing import_data: Dataframe is empty")
        raise err


def test_perform_eda():
    """
    Test the perform_eda function.

    Verifies that EDA images are created.

    Args:
        None

    Returns:
        None
    """
    try:
        df = cl.import_data("./data/bank_data.csv")
        cl.perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
        assert os.path.exists("./images/Churn_distribution.png")
        logging.info("Testing perform_eda: Images created successfully")
    except Exception as err:
        logging.error(f"Testing perform_eda: {err}")
        raise err


def test_encoder_helper():
    """
    Test the encoder_helper function.

    Verifies that encoded columns are added to the dataframe.

    Args:
        None

    Returns:
        None
    """
    try:
        df = cl.import_data("./data/bank_data.csv")
        category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
        response = 'Churn'
        df = cl.encoder_helper(df, category_lst, response)
        logging.info("Testing encoder_helper: SUCCESS")
        for category in category_lst:
            assert f"{category}_Churn" in df.columns
        logging.info("Testing encoder_helper: Encoded columns added successfully")
    except Exception as err:
        logging.error(f"Testing encoder_helper: {err}")
        raise err


def test_perform_feature_engineering():
    """
    Test the perform_feature_engineering function.

    Verifies that the data is split correctly.

    Args:
        None

    Returns:
        None
    """
    try:
        df = cl.import_data("./data/bank_data.csv")
        response = 'Churn'
        X_train, X_test, y_train, y_test = cl.perform_feature_engineering(df, response)
        logging.info("Testing perform_feature_engineering: SUCCESS")
        assert X_train.shape[0] > 0 and X_test.shape[0] > 0
        assert y_train.shape[0] > 0 and y_test.shape[0] > 0
        logging.info("Testing perform_feature_engineering: Data split successfully")
    except Exception as err:
        logging.error(f"Testing perform_feature_engineering: {err}")
        raise err


def test_train_models():
    """
    Test the train_models function.

    Verifies that models and related outputs are saved.

    Args:
        None

    Returns:
        None
    """
    try:
        df = cl.import_data("./data/bank_data.csv")
        response = 'Churn'
        X_train, X_test, y_train, y_test = cl.perform_feature_engineering(df, response)
        cl.train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
        assert os.path.exists("./models/logistic_model.pkl")
        assert os.path.exists("./models/rfc_model.pkl")
        assert os.path.exists("./images/logistic_regression_report.png")
        assert os.path.exists("./images/random_forest_report.png")
        assert os.path.exists("./images/rf_feature_importance.png")
        logging.info("Testing train_models: Models and reports saved successfully")
    except Exception as err:
        logging.error(f"Testing train_models: {err}")
        raise err


if __name__ == "__main__":
    """
    Main function to execute all test cases.

    Logs test results to churn_library.log and ensures all outputs
    of churn_library functions are tested for correctness.

    Args:
        None

    Returns:
        None
    """
    try:
        os.makedirs("./logs", exist_ok=True)
        test_import()
        test_perform_eda()
        test_encoder_helper()
        test_perform_feature_engineering()
        test_train_models()
        logging.info("All tests completed successfully")
    except Exception as e:
        logging.error(f"Error during testing: {e}")