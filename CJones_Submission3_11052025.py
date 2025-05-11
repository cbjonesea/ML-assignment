#!/usr/bin/env python
# coding: utf-8

# In[3]:


# --- Core ---
import pandas as pd
import numpy as np
import logging
import zipfile
import json 
import os


# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Scikit-learn: Preprocessing ---
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# --- Scikit-learn: Models ---
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor
)

# --- Scikit-learn: Metrics ---
from sklearn.metrics import mean_absolute_error

# --- Other ML Libraries ---
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

# --- Optional: Feature Engineering Tools ---
# from sklearn.cluster import KMeans
# from geopy.distance import geodesic

# --- Warnings ---
import warnings
warnings.filterwarnings('ignore')


# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load train and test data
logging.info("Loading train and test data...")
data = pd.read_json("train.json", orient="records")
test = pd.read_json("test.json", orient="records")

# Drop unused columns
drop_cols = ['host', 'name', 'facilities']
data.drop(columns=drop_cols, inplace=True)

# Domain-informed imputation: fill missing 'rooms' where room_type implies 1 room
room_fill_mask = data['room_type'].isin(['private_room', 'shared_room', 'hotel_room']) & data['rooms'].isna()
data.loc[room_fill_mask, 'rooms'] = 1 



# Split BEFORE any transformations
r_state = 123
raw_train, raw_valid = train_test_split(data, test_size=1/3, random_state=r_state)




# In[6]:


for i in [raw_train, raw_valid]:
    i.drop(labels=['guests','listing_type'], axis=1,inplace=True)


# In[8]:


raw_train['lat_lon_product'] = raw_train['lat'] * raw_train['lon']
raw_valid['lat_lon_product'] = raw_valid['lat'] * raw_valid['lon']


# In[10]:


# Feature engineering on raw_train and raw_valid
raw_train['rating_weighted'] = raw_train['rating'] * np.log1p(raw_train['num_reviews'])
raw_valid['rating_weighted'] = raw_valid['rating'] * np.log1p(raw_valid['num_reviews'])


# In[12]:


# Define target variable
target_col = 'revenue'

# Split raw_train into X and y
X_train = raw_train.drop(columns=[target_col])
y_train = raw_train[target_col]

# Split raw_valid into X and y
X_valid = raw_valid.drop(columns=[target_col])
y_valid = raw_valid[target_col]

# For log-transformed regression:
y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)  # (optional, if needed for eval)


# In[14]:


def make_preprocessor(imputation='simple', feature_set='basic'):
    if feature_set == 'basic':
        numeric_features = ['lat', 'lon', 'min_nights','rooms','bathrooms','beds','num_reviews','rating']
        categorical_features = ['room_type', 'cancellation']
    elif feature_set == 'engineered':
        numeric_features = ['lat', 'lon', 'min_nights','rooms','bathrooms','beds','num_reviews','rating', 'rating_weighted','lat_lon_product']
        categorical_features = ['room_type', 'cancellation']
    elif feature_set == 'engineered_reduced':
        numeric_features = ['min_nights','rooms','bathrooms','beds', 'rating_weighted','lat_lon_product']
        categorical_features = ['room_type', 'cancellation']
    else:
        raise ValueError("Invalid feature_set")

    if imputation == 'simple':
        num_imputer = SimpleImputer(strategy='mean')
    elif imputation == 'knn':
        num_imputer = KNNImputer(n_neighbors=5)
    elif imputation == 'iterative':
        num_imputer = IterativeImputer(random_state=0)
    else:
        raise ValueError("Invalid imputation type")

    numeric_pipeline = Pipeline(steps=[
        ('imputer', num_imputer),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    preprocessor_name = f"{imputation}_{feature_set}"
    return preprocessor, preprocessor_name


# In[15]:


def pipeline_iterator(model_type, imputation, feature_set, search_type=None, param_grid=None, n_iter=100, cv_folds=5):
    if search_type not in [None, 'grid', 'random']:
        raise ValueError("search_type must be None, 'grid', or 'random'.")

    if search_type in ['grid', 'random'] and param_grid is None:
        raise ValueError("param_grid must be provided for search_type='grid' or 'random'.")

    if search_type is None and param_grid is not None:
        logging.warning("param_grid is ignored since search_type is None.")

    # Instantiate model with random_state if possible
    try:
        model_instance = model_type(random_state=r_state)
    except TypeError:
        model_instance = model_type()

    # Define the pipeline
    preprocessor, preprocessor_name = make_preprocessor(imputation=imputation, feature_set=feature_set)
    logging.info(f"Using preprocessor: {preprocessor_name}")
    model_pipeline = make_pipeline(preprocessor, model_instance)


    # Select the estimator based on search_type
    if search_type == 'random':
        estimator = RandomizedSearchCV(
            estimator=model_pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='neg_mean_absolute_error',
            random_state=r_state,
            n_jobs=-1,
            verbose=1
        )
    elif search_type == 'grid':
        estimator = GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='neg_mean_absolute_error',
            verbose=1
        )
    else:
        estimator = model_pipeline

    logging.info(f"Fitting model: {model_type.__name__} using {preprocessor_name} and search_type: {search_type or 'none'}")
    estimator.fit(X_train, y_train_log)

    # If using CV, get the best estimator
    if search_type in ['grid', 'random']:
        best_model = estimator.best_estimator_
        logging.info(f"Best params: {estimator.best_params_}")
    else:
        best_model = estimator

    # Predictions
    train_preds_log = best_model.predict(X_train)
    valid_preds_log = best_model.predict(X_valid)

    train_preds = np.expm1(train_preds_log)
    valid_preds = np.expm1(valid_preds_log)

    train_mae = mean_absolute_error(raw_train['revenue'], train_preds)
    valid_mae = mean_absolute_error(y_valid, valid_preds)

    logging.info(f"{model_type.__name__} train MAE: {train_mae:.2f}")
    logging.info(f"{model_type.__name__} valid MAE: {valid_mae:.2f}")

    # return best_model

    return {
    'pipeline': estimator,  # this will be a full Pipeline object (or best_estimator_ if using Grid/RandomSearch)
    'train_mae': train_mae,
    'valid_mae': valid_mae,
    'mae_diff': abs(train_mae - valid_mae)
        }



# In[17]:


param_grid_lgbm = {
    'lgbmregressor__n_estimators': [100, 200, 300, 500],
    'lgbmregressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'lgbmregressor__max_depth': [-1, 4, 6, 8, 10],
    'lgbmregressor__num_leaves': [15, 31, 63, 127],
    'lgbmregressor__min_child_samples': [5, 10, 20, 50],
    'lgbmregressor__subsample': [0.6, 0.8, 1.0],
    'lgbmregressor__colsample_bytree': [0.6, 0.8, 1.0],
    'lgbmregressor__reg_alpha': [0.0, 0.1, 0.5, 1.0],
    'lgbmregressor__reg_lambda': [0.0, 0.1, 0.5, 1.0]
}

best_lgbm = pipeline_iterator(
            model_type=LGBMRegressor,
            imputation='simple',
            feature_set='engineered',
            search_type='random',
            param_grid=param_grid_lgbm,
            n_iter=100,
            cv_folds=5
        )


# In[18]:


## Recreate engineered feature in test data
test['lat_lon_product'] = test['lat'] * test['lon']
test['rating_weighted'] = test['rating'] * np.log1p(test['num_reviews'])

# Make predictions on the test set using the base model
pred_test = np.expm1(best_lgbm['pipeline'].predict(test))
test[target_col] = pred_test



# In[19]:


test.head()


# In[20]:


data['revenue'].describe()


# In[21]:


test['revenue'].describe()


# In[22]:


print(data['revenue'].median())
print(test['revenue'].median())


# In[23]:


print(data.shape)
print(test.shape)


# In[24]:


print(data.columns)
print(test.columns)


# In[25]:


# Convert the predictions to a list of dictionaries
predicted = test[[target_col]].to_dict(orient='records')


## Save the predictions to a ZIP file
with zipfile.ZipFile("cbjones_3_11052025.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    # Write the predictions to a JSON file inside the ZIP
    zipf.writestr("predicted.json", json.dumps(predicted, indent=2))


# In[26]:


# Get the current working directory
current_directory = os.getcwd()

# Path to the ZIP file (assuming it's in the current working directory)
zip_file_path = os.path.join(current_directory, 'cbjones_3_11052025.zip')

# Open the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all files to the current working directory
    zip_ref.extractall(current_directory)

print("Files extracted to:", current_directory)


# In[27]:


predicted=pd.read_json('predicted.json', orient='records')
predicted.shape

