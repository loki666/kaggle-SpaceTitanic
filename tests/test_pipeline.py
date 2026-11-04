from pathlib import Path

import numpy as np

from spaceship.features import CATEGORICAL, NUMERIC, TARGET, build_features
from spaceship.train import run

from synthetic import make_frame


def test_build_features_has_no_missing_values():
    train = make_frame(300, seed=0, with_target=True)
    test = make_frame(120, seed=1, with_target=False)
    ft, fte = build_features(train, test)
    assert len(ft) == 300 and len(fte) == 120
    assert ft[CATEGORICAL + NUMERIC].isna().sum().sum() == 0
    assert fte[CATEGORICAL + NUMERIC].isna().sum().sum() == 0
    assert set(ft[TARGET].unique()) <= {0, 1}
    assert TARGET not in fte.columns
    assert (fte["PassengerId"].to_numpy() == test["PassengerId"].to_numpy()).all()


def test_cryosleep_rules():
    train = make_frame(200, seed=2, with_target=True)
    test = make_frame(50, seed=3, with_target=False)
    ft, _ = build_features(train, test)
    # Nobody flagged as asleep may have spent anything.
    assert (ft.loc[ft["CryoSleep"] == 1, "TotalSpend"] == 0).all()


def test_run_writes_submission(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    make_frame(240, seed=4, with_target=True).to_csv(data / "train.csv", index=False)
    make_frame(60, seed=5, with_target=False).to_csv(data / "test.csv", index=False)
    out = tmp_path / "submission.csv"
    result = run(data, out, ["logreg", "lgbm"], folds=3, seed=0, report=tmp_path / "r.json")
    assert out.exists()
    lines = out.read_text().splitlines()
    assert lines[0] == "PassengerId,Transported"
    assert len(lines) == 61
    assert set(line.split(",")[1] for line in lines[1:]) <= {"True", "False"}
    assert set(result["cv_accuracy"]) == {"logreg", "lgbm", "blend"}
    assert all(0 <= v <= 1 for v in result["cv_accuracy"].values())
