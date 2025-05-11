# src/pipeline_util.py (FOR CITI BIKE PROJECT)

import lightgbm as lgb
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

# ==============================
# ✨ Manual Temporal Feature Engineering
# ==============================

def manual_temporal_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add manual temporal features to Citi Bike timeseries data.
    Assumes 'hour_ts' exists in X.
    """
    X = X.copy()

    if "hour_ts" in X.columns:
        # ✅ Safe check for datetime type (works with timezone-aware datetime too)
        if not pd.api.types.is_datetime64_any_dtype(X["hour_ts"]):
            X["hour_ts"] = pd.to_datetime(X["hour_ts"], utc=True)

        X["day"] = X["hour_ts"].dt.day
        X["week_of_year"] = X["hour_ts"].dt.isocalendar().week
        X["quarter"] = X["hour_ts"].dt.quarter
        X["is_start_of_month"] = (X["hour_ts"].dt.is_month_start).astype(int)
        X["is_end_of_month"] = (X["hour_ts"].dt.is_month_end).astype(int)

    return X

# ✅ Wrap into FunctionTransformer
add_manual_temporal_features = FunctionTransformer(manual_temporal_features, validate=False)

# ==============================
# 🛠 Drop Unnecessary Columns
# ==============================

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Drop unnecessary columns like 'hour_ts', 'start_station_name', 'time_of_day'.
    """

    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or ["hour_ts", "start_station_name", "time_of_day"]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        return X.drop(columns=self.columns_to_drop, errors="ignore")

# ✅ Instantiate
drop_unnecessary_columns = DropColumnsTransformer()

# ==============================
# 🚀 Final Citi Bike Pipeline
# ==============================

def get_pipeline(**hyper_params):
    """
    Returns a pipeline with manual feature engineering, column dropping,
    and LightGBM regressor for Citi Bike ride prediction.

    Parameters
    ----------
    **hyper_params : dict
        Parameters to pass to the LGBMRegressor.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    pipeline = make_pipeline(
        add_manual_temporal_features,
        drop_unnecessary_columns,
        lgb.LGBMRegressor(**hyper_params)
    )
    return pipeline
