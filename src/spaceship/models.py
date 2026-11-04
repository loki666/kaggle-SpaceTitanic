"""Model factories. Each returns an sklearn-compatible estimator that accepts
the raw feature frame produced by :mod:`spaceship.features`."""
from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from .features import CATEGORICAL, NUMERIC


def _ordinal_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ],
        verbose_feature_names_out=False,
    )


def make_lgbm(seed: int) -> Pipeline:
    clf = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        random_state=seed,
        verbose=-1,
    )
    return Pipeline([("prep", _ordinal_preprocessor()), ("clf", clf)])


def make_xgb(seed: int) -> Pipeline:
    clf = XGBClassifier(
        n_estimators=600,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        gamma=0.1,
        random_state=seed,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
    )
    return Pipeline([("prep", _ordinal_preprocessor()), ("clf", clf)])


class CatBoostWrapper(CatBoostClassifier):
    """CatBoost with native categorical handling on the raw frame."""

    def fit(self, X, y=None, **kwargs):  # noqa: N803
        return super().fit(X, y, cat_features=CATEGORICAL, **kwargs)


def make_catboost(seed: int) -> CatBoostWrapper:
    return CatBoostWrapper(
        iterations=1000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        thread_count=-1,
    )


def make_logreg(seed: int) -> Pipeline:
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    clf = LogisticRegression(C=0.5, max_iter=2000, random_state=seed)
    return Pipeline([("prep", prep), ("clf", clf)])


MODEL_FACTORIES = {
    "lgbm": make_lgbm,
    "xgb": make_xgb,
    "catboost": make_catboost,
    "logreg": make_logreg,
}
