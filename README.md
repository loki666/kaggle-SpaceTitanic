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
python -m spaceship.train --seeds 3                        # average 3 CV seeds
python -m spaceship.train --stack --pseudo-threshold 0.9   # see below
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
- Spatial interactions, on the hypothesis that the event was localised:
  deck x side, deck x side x 400-cabin band, home planet x deck, mean spend
  of the cabin and of neighbouring cabins (same deck/side, +/-5 numbers).
  Band width and radius are parameters of `build_features`.
- Behavioural proxy for where the passenger was when it happened: a
  location category (pod / cabin / spa-VR / venue / mixed / no spend)
  inferred from cryosleep and which services the spending went to, plus
  the log ratio of venue spend (food court, mall) to luxury spend
  (room service, spa, VR deck).
- Missing-value indicators for cabin, name and spending.

**Imputation** uses the structure of the data rather than global medians:
cryosleepers spend nothing (and spenders are awake); groups share a home
planet, destination and cabin; decks A/B/C/T are Europa-only and G is
Earth-only; age is filled by home planet and cryosleep status.

**Models** (`src/spaceship/models.py`): LightGBM, XGBoost, CatBoost (native
categoricals) and a one-hot logistic regression. Each is trained with
stratified K-fold; test predictions are averaged over folds, then the
models' probabilities are averaged and thresholded at 0.5.

Two optional extras, both evaluated so that the reported CV stays honest:

- `--pseudo-threshold T`: test rows the blend is confident about (prob
  above T or below 1-T) are added to training with their predicted label
  and every model is retrained. Done per fold, with the pseudo-labels
  coming from models that never saw that fold's validation rows.
- `--stack`: a logistic-regression meta-model over the models' out-of-fold
  logits replaces the plain average; its CV accuracy comes from a second
  10-fold cross-validation over the out-of-fold matrix.

## Results

5-fold stratified CV accuracy (out-of-fold, seed 42) and Kaggle public score.

| version | features                         | lgbm   | xgb    | catboost | logreg | blend  | public LB |
|---------|----------------------------------|--------|--------|----------|--------|--------|-----------|
| v1      | base                             | 0.8109 | 0.8121 | 0.8148   | 0.7971 | 0.8113 | 0.80266   |
| v3      | + spatial + behaviour            | 0.8171 | 0.8176 | 0.8179   | 0.8061 | 0.8193 | 0.80547   |
| v4      | v3, averaged over 3 CV seeds     | 0.8193 | 0.8159 | 0.8199   | 0.8058 | 0.8190 | 0.80523   |
| v5      | v3 + pseudo-label 0.9 + stack    | 0.8184 | 0.8162 | 0.8158   | 0.8047 | 0.8195 | 0.80570   |
| v6      | v3, neighbour radius 5, band 400 | 0.8215 | 0.8207 | 0.8165   | 0.8046 | 0.8226 | 0.80547   |
| v7      | v6, averaged over 3 CV seeds     | 0.8207 | 0.8207 | 0.8215   | 0.8036 | 0.8214 | 0.80640   |

Notes from the experiments (`--seeds 5`, LightGBM proxy):

- A leak-free "transported rate of training-fold neighbours within k cabins"
  feature adds nothing beyond cabin number itself, so the localisation is
  coarse (deck / side / band) rather than cabin-by-cabin.
- Hyperparameters move CV within about +/-0.003; CatBoost depth 8 was best.
- Blending beats the best single model by a hair; threshold 0.5 is optimal.
- Stacking (0.8187) does not beat the plain average (0.8193). Pseudo-labelling
  at thresholds 0.8 / 0.9 / 0.95 gives blends of 0.8186 / 0.8190 / 0.8190;
  with the stacker on top the best is 0.8195, two passengers better than the
  average and inside seed noise.
- Spatial sweep (10 seeds, mean of LightGBM and XGBoost): neighbour radius
  5 beats 20 by 0.0016 and the trend is monotone, so the useful neighbour
  signal is very local. Band width is flat between 50 and 1000; 400 was
  marginally best. Together: 0.8165 -> 0.8186 on the proxy.
- An MLP (scikit-learn stand-in for a Keras net) scores 0.803-0.804 alone and
  adds at most 0.0006 to the blend.

The data is synthetic with deliberate noise; honest models plateau around
0.82-0.83 on the public leaderboard. Higher public scores come from
leaderboard probing or shared test labels.

## Tests

```bash
pytest
```

Tests run the full pipeline on a small synthetic frame with the competition
schema. They only check that the code executes and produces a valid
submission; the synthetic data has no signal.
