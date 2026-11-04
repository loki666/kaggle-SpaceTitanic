"""Train a cross-validated ensemble and write a Kaggle submission.

Usage:
    python -m spaceship.train --data-dir data --out submissions/submission.csv
    python -m spaceship.train --stack --pseudo-threshold 0.9 --seeds 3

Each model is trained with stratified K-fold cross-validation. Out-of-fold
probabilities give an honest accuracy estimate per model and for the blend;
test predictions are the average over folds.

Options on top of the plain probability average:

* ``--pseudo-threshold T``: pseudo-labelling. Test rows whose blended
  probability is above T or below 1-T are added to the training data with
  their predicted label and every model is retrained. It is done per fold,
  with pseudo-labels produced by models that never saw that fold's
  validation rows, so the reported CV accuracy stays honest.
* ``--stack``: a logistic-regression meta-model over the models' out-of-fold
  probabilities replaces the plain average. Its CV accuracy is measured with
  a second cross-validation over the out-of-fold matrix.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .features import TARGET, build_features, feature_columns
from .models import MODEL_FACTORIES


def cv_predict(
    name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    folds: int,
    seed: int,
    pseudo: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (oof, test_by_fold) where test_by_fold has shape (folds, n_test).

    ``pseudo`` optionally gives, per fold, (test row indices, labels) to add to
    that fold's training data.
    """
    factory = MODEL_FACTORIES[name]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_by_fold = np.zeros((folds, len(X_test)))
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        X_fit, y_fit = X.iloc[tr], y[tr]
        if pseudo is not None:
            idx, labels = pseudo[fold]
            X_fit = pd.concat([X_fit, X_test.iloc[idx]], ignore_index=True)
            y_fit = np.concatenate([y_fit, labels])
        model = factory(seed + fold)
        model.fit(X_fit, y_fit)
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        test_by_fold[fold] = model.predict_proba(X_test)[:, 1]
    return oof, test_by_fold


def fit_models(models, X, y, X_test, folds, seed, seeds, pseudo=None, log=print):
    """Train every model over ``seeds`` CV seeds. Returns dicts of
    oof (n_train,) and test_by_fold (folds, n_test), averaged over seeds."""
    oofs, tests = {}, {}
    for name in models:
        t0 = time.time()
        oof = np.zeros(len(X))
        tbf = np.zeros((folds, len(X_test)))
        for s in range(seeds):
            o, t = cv_predict(name, X, y, X_test, folds, seed + 1000 * s, pseudo)
            oof += o / seeds
            tbf += t / seeds
        oofs[name], tests[name] = oof, tbf
        log(f"{name:>9s}  CV accuracy {accuracy_score(y, oof > 0.5):.4f}  ({time.time() - t0:.0f}s)")
    return oofs, tests


def select_pseudo(test_by_fold: np.ndarray, threshold: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per fold, the confident test rows (blended prob > T or < 1-T) and labels."""
    out = []
    for p in test_by_fold:
        idx = np.where((p >= threshold) | (p <= 1 - threshold))[0]
        out.append((idx, (p[idx] >= 0.5).astype(int)))
    return out


def stack(oofs: dict, tests: dict, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Logistic regression over the models' probabilities (logit space).
    Returns (oof_stacked, test_stacked, cv_accuracy_of_the_stacker)."""
    names = list(oofs)
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    M = np.column_stack([logit(oofs[n]) for n in names])
    M_test = np.column_stack([logit(tests[n].mean(axis=0)) for n in names])
    meta = LogisticRegression(C=1.0, max_iter=1000)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    oof_meta = cross_val_predict(meta, M, y, cv=skf, method="predict_proba")[:, 1]
    meta.fit(M, y)
    weights = dict(zip(names, np.round(meta.coef_[0], 3)))
    print(f"    stack  weights {weights}")
    return oof_meta, meta.predict_proba(M_test)[:, 1], accuracy_score(y, oof_meta > 0.5)


def run(
    data_dir: Path,
    out: Path,
    models: list[str],
    folds: int,
    seed: int,
    report: Path | None,
    seeds: int = 1,
    pseudo_threshold: float | None = None,
    use_stack: bool = False,
) -> dict:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    feat_train, feat_test = build_features(train, test)
    cols = feature_columns()
    X, y = feat_train[cols], feat_train[TARGET].to_numpy()
    X_test = feat_test[cols]

    oofs, tests = fit_models(models, X, y, X_test, folds, seed, seeds)
    scores = {n: accuracy_score(y, o > 0.5) for n, o in oofs.items()}
    blend_oof = np.mean(list(oofs.values()), axis=0)
    scores["blend"] = accuracy_score(y, blend_oof > 0.5)
    print(f"{'blend':>9s}  CV accuracy {scores['blend']:.4f}")

    if pseudo_threshold is not None:
        blend_tbf = np.mean(list(tests.values()), axis=0)
        pseudo = select_pseudo(blend_tbf, pseudo_threshold)
        n_pseudo = np.mean([len(p[0]) for p in pseudo])
        print(f"-- pseudo-labelling: {n_pseudo:.0f} of {len(X_test)} test rows per fold at threshold {pseudo_threshold}")
        oofs, tests = fit_models(models, X, y, X_test, folds, seed, seeds, pseudo=pseudo)
        for n, o in oofs.items():
            scores[f"{n}_pseudo"] = accuracy_score(y, o > 0.5)
        blend_oof = np.mean(list(oofs.values()), axis=0)
        scores["blend_pseudo"] = accuracy_score(y, blend_oof > 0.5)
        print(f"{'blend':>9s}  CV accuracy {scores['blend_pseudo']:.4f}")

    final_oof = blend_oof
    final_test = np.mean([t.mean(axis=0) for t in tests.values()], axis=0)
    if use_stack:
        final_oof, final_test, scores["stack"] = stack(oofs, tests, y, seed)
        print(f"{'stack':>9s}  CV accuracy {scores['stack']:.4f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"PassengerId": feat_test["PassengerId"], TARGET: final_test > 0.5})
    submission.to_csv(out, index=False)
    print(f"wrote {out} ({len(submission)} rows, {submission[TARGET].mean():.3f} transported)")

    result = {
        "models": models, "folds": folds, "seed": seed, "seeds": seeds,
        "pseudo_threshold": pseudo_threshold, "stack": use_stack,
        "cv_accuracy": {k: round(float(v), 5) for k, v in scores.items()},
        "final_cv_accuracy": round(float(accuracy_score(y, final_oof > 0.5)), 5),
    }
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
    p.add_argument("--pseudo-threshold", type=float, default=None, help="e.g. 0.9: pseudo-label confident test rows and retrain")
    p.add_argument("--stack", action="store_true", help="logistic-regression stacker over model probabilities")
    a = p.parse_args(argv)
    run(a.data_dir, a.out, a.models, a.folds, a.seed, a.report, a.seeds, a.pseudo_threshold, a.stack)


if __name__ == "__main__":
    main()
