#!/usr/bin/env python
# coding: utf-8

# In[7]:


## Import all needed libraries
import numpy as np
import pandas as pd
import json
import logging
import zipfile
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error


# In[8]:


# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Log the start of the process
logging.info("Reading train and test files")

# Read the training and test data from JSON files
train = pd.read_json("train.json", orient='records')
test = pd.read_json("test.json", orient='records')

# Split the data into training and validation sets before dropping columns
X = train.drop(labels=['host', 'name', 'listing_type', 'facilities', 'guests', #'bathrooms', 'beds', cancellation',
                       'revenue'], axis=1)
y = train['revenue']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=1/3, random_state=123)


# In[9]:


# Create list of transformations to apply in preprocessing
preprocess = ColumnTransformer(
    transformers=[
        # lat column: apply mean imputation and normalization
        ("lat", Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())
        ]), ["lat"]),
        # lon column: apply mean imputation and normalization
        ("lon", Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())
        ]), ["lon"]),
        # rooms column: apply median imputation (skewed data), log transformation (skewed data) and normalization
        ("rooms", Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('log', FunctionTransformer(np.log1p, validate=True)),
            ('scaler', MinMaxScaler())
        ]), ["rooms"]),
        # num_review column: apply median imputation (skewed data), log transformation (skewed data) and normalization
        ("num_reviews", Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('log', FunctionTransformer(np.log1p, validate=True)),
            ('scaler', MinMaxScaler())
        ]), ["num_reviews"]),
        # min_nights column: apply median imputation (skewed data), log transformation (skewed data) and normalization
        ("min_nights", Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('log', FunctionTransformer(np.log1p, validate=True)),
            ('scaler', MinMaxScaler())
        ]), ["min_nights"]),
        # rating column: apply median imputation (skewed data), log transformation (skewed data) and normalization
        ("rating", Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('log', FunctionTransformer(np.log1p, validate=True)),
            ('scaler', MinMaxScaler())
        ]), ["rating"]),
        # bathroom column: apply median imputation (skewed data), log transformation (skewed data) and normalization
        ("bathrooms", Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('log', FunctionTransformer(np.log1p, validate=True)),
            ('scaler', MinMaxScaler())
        ]), ["bathrooms"]),
        # beds column: apply median imputation (skewed data), log transformation (skewed data) and normalization
        ("beds", Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('log', FunctionTransformer(np.log1p, validate=True)),
            ('scaler', MinMaxScaler())
        ]), ["beds"]),
        # room_type column: apply one-hot encoding
        ("room_type", OneHotEncoder(handle_unknown='ignore'), ["room_type"]),
        # cancellation column: apply one-hot encoding
        ("cancellation", Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with the most frequent category
            ('onehot', OneHotEncoder(handle_unknown='ignore'))      # Apply one-hot encoding
        ]), ["cancellation"])
    ],
    remainder='drop'
)

# Fit the preprocessing steps on the training data
preprocess.fit(X_train)



# In[14]:


from sklearn.ensemble import GradientBoostingRegressor

# Create a Gradient Boosting Regressor pipeline including preprocessing steps
gradient_boosting = make_pipeline(preprocess, GradientBoostingRegressor(random_state=123))

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'gradientboostingregressor__n_estimators': [50, 100, 200],
    'gradientboostingregressor__learning_rate': [0.01, 0.1, 0.2],
    'gradientboostingregressor__max_depth': [3, 4, 5],
    'gradientboostingregressor__min_samples_split': [2, 5, 10],
    'gradientboostingregressor__min_samples_leaf': [1, 2, 4],
    'gradientboostingregressor__max_features': ['sqrt', 'log2', None],
    'gradientboostingregressor__subsample': [0.8, 0.9, 1.0]
}

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(gradient_boosting, param_distributions=param_grid, n_iter=50, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=4, random_state=123)

# Fit the RandomizedSearchCV object to the training data
random_search.fit(X_train, np.log1p(y_train.values))

# Get the best model from RandomizedSearchCV
best_gradient_boosting = random_search.best_estimator_

# Define the target label
label = 'revenue'

# Evaluate the best model on both training and validation sets
for split_name, X_split, y_split in [("train     ", X_train, y_train), ("valid     ", X_valid, y_valid)]:
    # Predict target values from the combined train and validation sets (X_split)
    pred = np.expm1(best_gradient_boosting.predict(X_split))

    # Calculate the mean absolute error
    mae = mean_absolute_error(y_split, pred)

    # Log the performance
    logging.info(f"Gradient Boosting {split_name} {mae:.3f}")

# Make predictions on the test set using the base model
pred_test = np.expm1(best_gradient_boosting.predict(test))
test[label] = pred_test

# Convert the predictions to a list of dictionaries
predicted = test[[label]].to_dict(orient='records')

## Save the predictions to a ZIP file
with zipfile.ZipFile("cbjones_2_30042025.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    # Write the predictions to a JSON file inside the ZIP
    zipf.writestr("predicted.json", json.dumps(predicted, indent=2))


# In[16]:


import os

# Get the current working directory
current_directory = os.getcwd()

# Path to the ZIP file (assuming it's in the current working directory)
zip_file_path = os.path.join(current_directory, 'cbjones_2_30042025.zip')

# Open the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all files to the current working directory
    zip_ref.extractall(current_directory)

print("Files extracted to:", current_directory)


# In[18]:


predicted=pd.read_json('predicted.json', orient='records')
predicted.shape

