import sys
import os

# Add the current working directory to Python path
sys.path.append(os.getcwd())

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

#from src.feature_engineering import TransactionTimeFeatures
from src.feature_engineering import TransactionTimeFeatures

#import sys
#sys.path.append('./src')
df = pd.read_csv('C:\\Users\\Hiwi\\Documents\\week5\\data.csv')
# ================================
# Define column groups
# ================================

CATEGORICAL_COLS = [
    'CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy'
]

NUMERICAL_COLS = [
    'Amount', 'Value'
]

DATETIME_COL = 'TransactionStartTime'

# ================================
# Pipelines for sub-transforms
# ================================

# Pipeline for categorical features
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Pipeline for numerical features
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# ================================
# Preprocessing ColumnTransformer
# ================================

def build_transaction_pipeline():
    preprocessing = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, NUMERICAL_COLS),
        ('cat', categorical_pipeline, CATEGORICAL_COLS)
    ])

    # Final pipeline with datetime features first
    full_pipeline = Pipeline(steps=[
        ('datetime_features', TransactionTimeFeatures(datetime_col=DATETIME_COL)),
        ('preprocessing', preprocessing)
    ])

    return full_pipeline
