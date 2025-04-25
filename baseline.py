import logging
import pandas as pd
import numpy as np
import json
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error


def baseline():
    logging.info("Reading train and test files")
    train = pd.read_json("train.json", orient='records')
    test = pd.read_json("test.json", orient='records')
    train, valid = train_test_split(train, test_size=1/3, random_state=123)
    preprocess = ColumnTransformer(
        transformers=[
            ("lat", Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  
            ]), ["lat"]),
            ("lon", Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  
            ]), ["lon"]),
            ("rooms", Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),  
            ]), ["rooms"]),
            ("num_reviews", Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),  
            ]), ["num_reviews"]),
        ],
        remainder='drop'
    )
    dummy = make_pipeline(preprocess, DummyRegressor())
    base = make_pipeline(preprocess, Ridge(alpha=1, random_state=123))
    label = 'revenue'
    for model_name, model in [("mean", dummy),
                              ("base", base)
                        ]:
        
        logging.info(f"Fitting model {model_name}")
        model.fit(train.drop([label], axis=1), np.log1p(train[label].values))
        for split_name, split in [("train     ", train),
                                  ("valid     ", valid)
                                  ]:
            pred = np.expm1(model.predict(split.drop([label], axis=1)))
            mae = mean_absolute_error(split[label], pred)
            logging.info(f"{model_name} {split_name} {mae:.3f}")
    pred_test = np.expm1(base.predict(test))
    test[label] = pred_test
    predicted = test[['revenue']].to_dict(orient='records')
    with zipfile.ZipFile("baseline.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        # Write to a file inside the ZIP
        zipf.writestr("predicted.json", json.dumps(predicted, indent=2))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    baseline()
