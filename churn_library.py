"""
Function library for data processing, exploratory data analysis, result ploting , feature engineering and model training.

Author : TC KONG
Creation date : 5/12/2024
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
import joblib

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    """
    Reads a CSV file and returns it as a pandas DataFrame.

    Args:
        pth (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(pth)


def perform_eda(df):
    """
    Performs exploratory data analysis (EDA) on a DataFrame and saves figures.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        None
    """
    os.makedirs('images', exist_ok=True)
    
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(f'images/{col}_distribution.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Box plot of {col}')
        plt.savefig(f'images/{col}_boxplot.png')
        plt.close()

    for col in df.select_dtypes(include=['object']).columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[col])
        plt.title(f'Count plot of {col}')
        plt.savefig(f'images/{col}_countplot.png')
        plt.close()


def encoder_helper(df, category_lst, response):
    """
    Encodes categorical columns with the mean response for each category.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        category_lst (list): List of categorical column names.
        response (str): Response column name.

    Returns:
        pandas.DataFrame: Updated DataFrame.
    """
    for cat in category_lst:
        cat_churn_mean = df.groupby(cat)[response].mean()
        df[f'{cat}_Churn'] = df[cat].map(cat_churn_mean)
    return df


def perform_feature_engineering(df, response):
    """
    Splits the DataFrame into training and testing datasets.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        response (str): Response column name.

    Returns:
        tuple: Split datasets (X_train, X_test, y_train, y_test).
    """
    y = df[response]
    X = df.drop(columns=[response])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf):
    """
    Generates classification reports and saves them as images.

    Args:
        y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf:
            Actual and predicted values for training and testing.

    Returns:
        None
    """
    os.makedirs('images', exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('images/logistic_regression_report.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('images/random_forest_report.png')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    """
    Creates and saves a feature importance plot.

    Args:
        model: Trained model with feature_importances_ attribute.
        X_data (pandas.DataFrame): DataFrame of features.
        output_pth (str): Path to save the plot.

    Returns:
        None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.barh(range(X_data.shape[1]), importances[indices], align="center")
    plt.yticks(range(X_data.shape[1]), names)
    plt.savefig(output_pth)
    plt.close()


def plot_roc_curve(y_test, preds, model_name):
    """
    Plots and saves the ROC curve.

    Args:
        y_test (array): True labels.
        preds (array): Predicted probabilities.
        model_name (str): Name of the model for labeling.

    Returns:
        None
    """
    fpr, tpr, _ = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'images/roc_curve_{model_name}.png')
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    """
    Trains models, generates and saves results.

    Args:
        X_train, X_test, y_train, y_test: Split datasets.

    Returns:
        None
    """
    os.makedirs('models', exist_ok=True)

    lr = LogisticRegression()
    rf = RandomForestClassifier()

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    y_train_preds_lr = lr.predict(X_train)
    y_test_preds_lr = lr.predict_proba(X_test)[:, 1]  # Probabilities for ROC

    y_train_preds_rf = rf.predict(X_train)
    y_test_preds_rf = rf.predict_proba(X_test)[:, 1]  # Probabilities for ROC

    classification_report_image(
        y_train, y_test, y_train_preds_lr, y_train_preds_rf,
        lr.predict(X_test), rf.predict(X_test)
    )

    plot_roc_curve(y_test, y_test_preds_lr, 'Logistic Regression')
    plot_roc_curve(y_test, y_test_preds_rf, 'Random Forest')

    joblib.dump(lr, 'models/logistic_model.pkl')
    joblib.dump(rf, 'models/rfc_model.pkl')

    feature_importance_plot(rf, X_train, 'images/rf_feature_importance.png')
