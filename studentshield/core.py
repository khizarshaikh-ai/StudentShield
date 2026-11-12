
from __future__ import annotations

import io
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
)


# --------------------------------------------------
# CSV READER
# --------------------------------------------------

def read_student_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    errors = []

    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep)

            if df.shape[1] > 1:
                return df

        except Exception as exc:
            errors.append(str(exc))

    raise ValueError(
        "Could not parse CSV. Check the file format and delimiter. "
        + "; ".join(errors)
    )


# --------------------------------------------------
# DATA PREPARATION + DIAGNOSTICS
# --------------------------------------------------

def prepare_xy_diagnostic(
    df,
    target,
    task,
    selected_features=None
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' does not exist."
        )

    if task not in ("Regression", "Grade bands"):
        raise ValueError(f"Unsupported task: {task}")

    if selected_features is None:
        features = [
            c for c in df.columns if c != target
        ]
    else:
        features = list(dict.fromkeys(
            c for c in selected_features
            if c in df.columns and c != target
        ))

    if not features:
        raise ValueError(
            "Select at least one valid input feature."
        )

    raw_rows = len(df)

    # Convert target to numeric.
    numeric_target = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    # Exclude missing and infinite target values.
    finite_target = pd.Series(
        np.isfinite(
            numeric_target.to_numpy(dtype=float)
        ),
        index=df.index
    )

    valid = numeric_target.notna() & finite_target

    if task == "Grade bands":
        valid &= numeric_target.between(0, 20)

    valid_positions = np.flatnonzero(
        valid.to_numpy()
    )

    X = df.iloc[valid_positions][features].copy()

    y_numeric = numeric_target.iloc[
        valid_positions
    ].astype(float)

    if task == "Grade bands":
        y = pd.cut(
            y_numeric,
            bins=[-0.1, 9.5, 13.5, 20.1],
            labels=["low", "middle", "high"]
        )

        valid_bands = y.notna()

        X = X.loc[valid_bands]
        y = y.loc[valid_bands].astype(str)

    else:
        y = y_numeric

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if len(X) == 0:
        raise ValueError(
            f"No usable rows for target '{target}'. "
            "Choose a valid numeric target, such as G3."
        )

    if X.shape[1] == 0:
        raise ValueError(
            "No input features remain after preparation."
        )

    missing_target_rows = int(
        df[target].isna().sum()
    )

    diagnostics = {
        "raw_rows": int(raw_rows),
        "missing_target_rows": missing_target_rows,
        "invalid_target_rows": int(
            raw_rows - len(valid_positions)
            - missing_target_rows
        ),
        "usable_rows": int(len(X)),
        "dropped_rows": int(raw_rows - len(X)),
        "target": target,
        "task": task,
        "grade_distribution": (
            y.value_counts().to_dict()
            if task == "Grade bands"
            else {}
        ),
        "feature_missing_cells": int(
            X.isna().sum().sum()
        ),
        "feature_count": int(X.shape[1]),
    }

    return X, y, features, diagnostics


def prepare_xy(
    df,
    target,
    task,
    selected_features=None
):
    X, y, features, _ = prepare_xy_diagnostic(
        df,
        target,
        task,
        selected_features
    )

    return X, y, features


# --------------------------------------------------
# XGBOOST CLASS LABEL WRAPPER
# --------------------------------------------------

class EncodedXGBClassifier(ClassifierMixin, BaseEstimator):
    """
    Encodes string class labels before fitting XGBoost.
    Decodes predictions back to the original labels.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()

        encoded_y = self.label_encoder_.fit_transform(
            np.asarray(y)
        )

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, encoded_y)

        self.classes_ = self.label_encoder_.classes_

        return self

    def predict(self, X):
        encoded_predictions = self.estimator_.predict(X)

        return self.label_encoder_.inverse_transform(
            np.asarray(encoded_predictions, dtype=int)
        )

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


# --------------------------------------------------
# MODEL PIPELINE
# --------------------------------------------------

def make_pipeline(
    X,
    task,
    model_name,
    random_state=42
):
    if task not in ("Regression", "Grade bands"):
        raise ValueError(f"Unsupported task: {task}")

    if model_name not in ("Random Forest", "XGBoost"):
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    if X.empty or X.shape[1] == 0:
        raise ValueError(
            "Cannot train a model without input features."
        )

    categorical = list(
        X.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns
    )

    numeric = [
        c for c in X.columns
        if c not in categorical
    ]

    transformers = []

    if numeric:
        num_pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True
                )
            )
        ])

        transformers.append(
            ("num", num_pipe, numeric)
        )

    if categorical:
        cat_pipe = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ])

        transformers.append(
            ("cat", cat_pipe, categorical)
        )

    prep = ColumnTransformer(
        transformers,
        remainder="drop"
    )

    is_regression = task == "Regression"

    if model_name == "XGBoost":
        try:
            from xgboost import (
                XGBRegressor,
                XGBClassifier
            )

        except ImportError:
            raise RuntimeError(
                "XGBoost is not installed. "
                "Run: pip install xgboost"
            )

        if is_regression:
            estimator = XGBRegressor(
                n_estimators=180,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=2
            )

        else:
            base_estimator = XGBClassifier(
                n_estimators=180,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=2
            )

            estimator = EncodedXGBClassifier(
                estimator=base_estimator
            )

    else:
        if is_regression:
            estimator = RandomForestRegressor(
                n_estimators=180,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            )

        else:
            estimator = RandomForestClassifier(
                n_estimators=180,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1
            )

    return Pipeline([
        ("preprocess", prep),
        ("model", estimator)
    ])


# --------------------------------------------------
# PER-RECORD LOSS
# --------------------------------------------------

def per_record_loss(model, X, y, task):
    if task == "Regression":
        predictions = np.asarray(
            model.predict(X),
            dtype=float
        )

        actual = np.asarray(
            y,
            dtype=float
        )

        return np.square(actual - predictions)

    probabilities = model.predict_proba(X)

    estimator = model.named_steps["model"]

    classes = list(estimator.classes_)

    class_to_idx = {
        label: index
        for index, label in enumerate(classes)
    }

    y_arr = np.asarray(y)

    eps = 1e-12
    losses = []

    for i, actual_class in enumerate(y_arr):
        if actual_class not in class_to_idx:
            # The model did not observe this class.
            # Assign a very high loss.
            probability = eps
        else:
            probability = float(
                probabilities[
                    i,
                    class_to_idx[actual_class]
                ]
            )

        losses.append(
            -np.log(max(probability, eps))
        )

    return np.asarray(losses, dtype=float)


# --------------------------------------------------
# TARGET MODEL TRAINING
# --------------------------------------------------

def train_target(
    X,
    y,
    task,
    model_name,
    test_size=0.30,
    seed=42
):
    from sklearn.model_selection import train_test_split

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if len(X) < 2:
        raise ValueError(
            "At least two usable records are required."
        )

    if len(X) != len(y):
        raise ValueError(
            "Feature and target row counts do not match."
        )

    idx = np.arange(len(X))

    stratify = None

    if task == "Grade bands":
        counts = y.value_counts()
        n_classes = len(counts)

        n_audit = int(
            np.ceil(len(X) * test_size)
        )

        n_train = len(X) - n_audit

        if n_classes < 2:
            raise ValueError(
                "Grade-band classification requires "
                "at least two classes."
            )

        if (
            counts.min() < 2
            or n_audit < n_classes
            or n_train < n_classes
        ):
            raise ValueError(
                "Not enough records for a safe stratified split. "
                "Collect more examples from each grade band."
            )

        stratify = y

    train_idx, audit_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=stratify
    )

    model = make_pipeline(
        X.iloc[train_idx],
        task,
        model_name,
        seed
    )

    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx]
    )

    losses = per_record_loss(
        model,
        X.iloc[audit_idx],
        y.iloc[audit_idx],
        task
    )

    return model, train_idx, audit_idx, losses
