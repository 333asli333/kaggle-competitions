# Spaceship Titanic

Kaggle competition: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)

**Question:** which passengers were transported to another dimension by the anomaly, and can a model learn why?

| Notebook | Language |
|---|---|
| [`spaceship_titanic.ipynb`](spaceship_titanic.ipynb) | English |
| [`spaceship_titanic_tr.ipynb`](spaceship_titanic_tr.ipynb) | Türkçe |

## Story

| Chapter | Question |
|---|---|
| 1 | What is in the data, and where are the gaps? |
| 2 | Who was transported? |
| 3 | How is the data prepared for a model? |
| 4 | Which findings turn into features? |
| 5 | Which model is better? |
| 6 | What do tuning and ensembling add? |
| 7 | Where does the model get it wrong? |

## Key findings

- **Sleep and spending are the strongest clue:** 82% of the passengers in cryosleep were transported, against 30% of those who spent money on board.
- **The kind of spending matters:** among passengers spending more than 3000 on room service, spa and the VR deck the rate drops to 5%. For the food court and shopping mall the relation is U-shaped.
- **Location decides:** decks B and C 68–73%, starboard side 56%, port side 45%. On the lower decks, starboard cabins 600–1199 reach 71%.
- **Home planet is entangled with the deck:** Europa 66%, Earth 42%. Of the passengers in cryosleep, 99% of the Europans were transported but only 66% of the Earthlings.
- **Young children are a group of their own:** 77% for ages 0–4, 46–50% for adults.
- **Groups did not share a fate:** solo travellers 45%, group travellers 57%; yet in only 43.6% of groups did every member share the same outcome (38.4% with shuffled labels).
- **Missingness is random:** passengers with and without missing values were transported at the same rate.

## Method

- **Rule-based imputation:** 12 rules, 73–100% accurate (a sleeping passenger spends 0, the surname gives the home planet, the group gives the cabin). Remaining gaps are filled inside a leak-free `Pipeline`, with statistics learned from the training fold only.
- **Cross-validation:** StratifiedGroupKFold, 5 folds × 3 repeats. Groups are never split between train and test on Kaggle, so they are not split in CV either.
- **Comparison:** every experiment runs on the same 15 splits; differences are checked with the corrected paired t-test (Nadeau & Bengio).
- **Tuning:** Optuna searches on a separate CV split and results are reported on independent splits.

## Result

| Model | CV accuracy | Kaggle public |
|---|---|---|
| Simple rule: no spending means transported | 73.7% | – |
| Logistic Regression | 79.9% | – |
| HistGradientBoosting + spending groups | 81.3% | 0.80219 |
| CatBoost, tuned with Optuna | 81.6% | 0.80944 |
| **Ensemble: 4 tuned gradient boosting models** | **81.5%** | **0.81038** |
| Ensemble + `HotZone` | 81.5% | 0.80851 |

## What we learned

- **The score was decided by whether the information given to the model was new, not by how the model was tuned.** Rule-based imputation (73–100% accurate), the Optuna search, the ensemble and the cabin region found during exploration: none of them moved the score by a meaningful margin. The only feature that helped was splitting spending into luxury and basic services.
- **The search showed +0.3 points on its own CV split, and that gain vanished on independent splits.** Searching on a separate split is what made the illusion visible.
- **76% of the errors are Earthlings, mostly on deck G.** Between 52% and 66% of them were transported, and nothing in the data tells them apart.

## Competition rules

- **Task / metric:** binary classification, accuracy.
- **Submission:** `PassengerId,Transported` (True/False), 4277 rows. At most **10** submissions per day.
- **Leaderboard:** the public / private split of the test set is hidden; submissions older than two months drop off. No end date.
- **Teams:** at most 10 members, one account per participant. Code may be shared inside a team or publicly (forum / notebooks), never privately.
- **Data:** not redistributable (kept out of this repository). External data is allowed if it is public and free for everyone. Hand-labelling the test data is forbidden. AutoML is allowed.

## Pipeline

| # | Step | Contents | Status |
|---|---|---|---|
| 1 | Knowing the data | Rows and columns, what each column means, data types and conversion, duplicate records, missing values (drop or fill?), basic statistics (mean, median, spread) | ✅ |
| 2 | Exploratory analysis | Each variable against the target: CryoSleep, spending, cabin, group/family, home planet, age | ✅ |
| 3 | Preprocessing | Splitting composite columns (`PassengerId`, `Cabin`, `Name`), rule-based imputation, encoding, scaling; all inside a leak-free `Pipeline` | ✅ |
| 4 | Feature engineering | Group size, total spending, family, cabin region | ✅ |
| 5 | Models | LR, RF, HistGB, XGBoost, LightGBM, CatBoost on identical folds, compared statistically | ✅ |
| 6 | Tuning and ensembling | Optuna, blending / stacking | ✅ |
| 7 | Submissions and error analysis | Selection by CV, checked against the leaderboard, the `HotZone` experiment | ✅ |

## Experiment log

| # | Change | Accuracy | F1 | ROC AUC | Public LB |
|---|---|---|---|---|---|
| 1 | Logistic Regression, base columns, rule-based filling | 77.81 ± 0.84 | 78.26 | 85.30 | |
| 2 | HistGradientBoosting, base columns, no rules | 80.85 ± 0.63 | 80.98 | 90.03 | |
| 3 | HistGradientBoosting, base columns, rule-based filling | 80.75 ± 0.81 | 80.89 | 90.22 | |
| 4 | HistGB + spending groups → `01_hgb_spend_groups.csv` | 81.26 ± 0.81 | 81.41 | 90.49 | 80.22 |
| 5 | HistGB + all new features | 81.14 ± 0.64 | 81.25 | 90.51 | |
| 6 | Logistic Regression + all new features | 79.89 ± 0.91 | 80.29 | 86.87 | |
| 7 | Random Forest + all features | 80.69 ± 0.62 | 80.63 | 89.90 | |
| 8 | XGBoost + all features | 81.07 ± 0.52 | 81.21 | 90.50 | |
| 9 | LightGBM + all features | 81.23 ± 0.69 | 81.20 | 90.47 | |
| 10 | CatBoost + all features | 81.53 ± 0.90 | 81.77 | 90.84 | |
| 11 | CatBoost, tuned with Optuna → `02_catboost_tuned.csv` | 81.59 ± 0.80 | 81.85 | 90.83 | 80.94 |
| 12 | LightGBM / HistGB / XGBoost, tuned | 81.16–81.32 | 81.55–81.68 | 90.49–90.68 | |
| 13 | Ensemble: average of 4 tuned boosting models → `03_ensemble_4_boosting.csv` | 81.51 ± 0.78 | 81.83 | 90.81 | **81.04** |
| 14 | Ensemble + `HotZone` (decks F/G, starboard, 600–1199) → `04_ensemble_hotzone.csv` | 81.54 ± 0.73 | 81.85 | 90.86 | 80.85 |

CV: StratifiedGroupKFold, 5 folds × 3 repeats. The Optuna settings live in `tuning/best_params.json`; the notebook reads them instead of running the search again.

## Run

Download the competition data into `dataset/` (`train.csv`, `test.csv`, `sample_submission.csv`). On Kaggle the notebook finds the data under `/kaggle/input` automatically.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook spaceship_titanic.ipynb
```

On macOS, XGBoost and LightGBM need the OpenMP runtime (`brew install libomp`). Without Homebrew, the copy shipped inside scikit-learn can be linked instead:

```bash
SP=.venv/lib/python3.9/site-packages
for lib in $SP/xgboost/lib/libxgboost.dylib $SP/lightgbm/lib/lib_lightgbm.dylib; do
  ln -sf ../../sklearn/.dylibs/libomp.dylib "$(dirname $lib)/libomp.dylib"
  install_name_tool -add_rpath @loader_path "$lib"
  codesign --force -s - "$lib"
done
```
