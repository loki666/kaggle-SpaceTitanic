"""Train a cross-validated ensemble and write a Kaggle submission.

Usage:
    python -m spaceship.train --data-dir data --out submissions/submission.csv

Each model is trained with stratified K-fold cross-validation. Out-of-fold
probabilities give an honest accuracy estimate per model and for the blend;
test predictions are the average over folds. The final prediction is the
mean probability across the selected models thresholded at 0.5.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from .features import TARGET, build_features, feature_columns
from .models import MODEL_FACTORIES


def cv_predict(name: str, X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, folds: int, seed: int):
    factory = MODEL_FACTORIES[name]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = factory(seed + fold)
        model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / folds
    return oof, test_pred


def run(data_dir: Path, out: Path, models: list[str], folds: int, seed: int, report: Path | None, seeds: int = 1) -> dict:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    feat_train, feat_test = build_features(train, test)
    cols = feature_columns()
    X, y = feat_train[cols], feat_train[TARGET].to_numpy()
    X_test = feat_test[cols]

    scores: dict[str, float] = {}
    oofs, tests = [], []
    for name in models:
        t0 = time.time()
        oof = np.zeros(len(X))
        test_pred = np.zeros(len(X_test))
        for s in range(seeds):
            o, t = cv_predict(name, X, y, X_test, folds, seed + 1000 * s)
            oof += o / seeds
            test_pred += t / seeds
        scores[name] = accuracy_score(y, oof > 0.5)
        oofs.append(oof)
        tests.append(test_pred)
        print(f"{name:>9s}  CV accuracy {scores[name]:.4f}  ({time.time() - t0:.0f}s)")

    blend_oof = np.mean(oofs, axis=0)
    blend_test = np.mean(tests, axis=0)
    scores["blend"] = accuracy_score(y, blend_oof > 0.5)
    print(f"{'blend':>9s}  CV accuracy {scores['blend']:.4f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"PassengerId": feat_test["PassengerId"], TARGET: blend_test > 0.5})
    submission.to_csv(out, index=False)
    print(f"wrote {out} ({len(submission)} rows, {submission[TARGET].mean():.3f} transported)")

    result = {"models": models, "folds": folds, "seed": seed, "seeds": seeds, "cv_accuracy": scores}
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("submissions/submission.csv"))
    p.add_argument("--models", nargs="+", default=["lgbm", "xgb", "catboost", "logreg"], choices=list(MODEL_FACTORIES))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report", type=Path, default=Path("submissions/cv_report.json"))
    p.add_argument("--seeds", type=int, default=1, help="average over this many CV seeds")
    a = p.parse_args(argv)
    run(a.data_dir, a.out, a.models, a.folds, a.seed, a.report, a.seeds)


if __name__ == "__main__":
    main()
