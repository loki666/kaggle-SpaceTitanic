# Spaceship Titanic (Kaggle)

Pipeline for the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
competition: predict which passengers were transported to another dimension.

## Setup

```bash
pip install -r requirements.txt
pip install -e .          # makes `python -m spaceship.train` importable from anywhere
scripts/download_data.sh  # needs Kaggle credentials + accepted competition rules
```

`data/` is git-ignored; the download script places `train.csv`, `test.csv`
and `sample_submission.csv` there.

## Train and predict

```bash
python -m spaceship.train                       # all four models, 5-fold CV
python -m spaceship.train --models lgbm catboost --folds 10 --seed 7
```

Prints the out-of-fold accuracy per model and for the blend, writes
`submissions/submission.csv` and `submissions/cv_report.json`.

```bash
scripts/submit.sh submissions/submission.csv "lgbm+xgb+catboost+logreg blend"
```

## What the pipeline does

**Features** (`src/spaceship/features.py`), computed on train and test
together so group statistics see the whole manifest:

- `PassengerId` -> travel group, group size, alone flag.
- `Cabin` -> deck, cabin number, side, cabin region bucket, cabin-mate count.
- `Name` -> surname and family size.
- Spending: total, log total, luxury (RoomService/Spa/VRDeck) vs basic
  (FoodCourt/ShoppingMall), count of non-zero categories, no-spend flag.
- Group aggregates: mean spend, cryosleep rate, no-spend rate.
- Missing-value indicators for age, cabin, name and spending.

**Imputation** uses the structure of the data rather than global medians:
cryosleepers spend nothing (and spenders are awake); groups share a home
planet, destination and cabin; decks A/B/C/T are Europa-only and G is
Earth-only; age is filled by home planet and cryosleep status.

**Models** (`src/spaceship/models.py`): LightGBM, XGBoost, CatBoost (native
categoricals) and a one-hot logistic regression. Each is trained with
stratified K-fold; test predictions are averaged over folds, then the
models' probabilities are averaged and thresholded at 0.5.

## Tests

```bash
pytest
```

Tests run the full pipeline on a small synthetic frame with the competition
schema. They only check that the code executes and produces a valid
submission; the synthetic data has no signal.
