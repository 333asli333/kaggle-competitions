# Titanic: The Story in the Data

Kaggle competition: [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

**Question:** who survived the sinking of the Titanic, and can a model learn why?

| Notebook | Language |
|---|---|
| [`titanic.ipynb`](titanic.ipynb) | English |
| [`titanic_tr.ipynb`](titanic_tr.ipynb) | Türkçe |

## Story

| Chapter | Question |
|---|---|
| 1 | Who was aboard? |
| 2 | Who survived? |
| 3 | Whose records are missing? |
| 4 | Can a model learn the story? |
| 5 | Where does the model fail? |
| 6 | What does it predict for the 418 hidden passengers? |

## Key findings

- **Sex made the biggest difference:** 74% of women and 19% of men survived.
- **Class changed the pattern:** a 3rd class woman (50%) survived more often than a 1st class man (37%).
- **Title separates boys from men:** Master 57% versus Mr 16%.
- **Group size is an inverted U:** alone 27%, small group 55%, large group 28%.
- **Missing records are not random:** no recorded age for 64% of Queenstown passengers.

## Modelling

- Features: `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `Title`, `GroupSize` (family and ticket groups), `CabinShare`, `AgeMissing`.
- Leak-free pipeline: fill values are learned inside cross-validation.
- 8 models compared on the same 15 folds (5 folds × 3 repeats), checked with a corrected resampled t-test, a Bayesian correlated t-test, McNemar and Friedman/Nemenyi.
- Hyperparameters tuned with nested cross-validation.

| Model (nested CV) | Accuracy |
|---|---|
| Gradient Boosting | 83.7% |
| **Logistic Regression + LogFare** (final) | **83.1%** |
| XGBoost | 82.8% |

Gradient Boosting and Logistic Regression are statistically equivalent, so the simpler, interpretable model was chosen.

## Result

| Submission | Kaggle public score |
|---|---|
| Kaggle sample file: women survive, men die | 0.76555 ([source](https://www.kaggle.com/mylesoneill/tutorial-1-gender-based-model-0-76555)) |
| **This notebook** | **0.76794** |

The model beat the simple rule by a clear margin in cross-validation, but on Kaggle's hidden passengers only by one passenger. **Good, but not good enough:** the extra rules beyond sex did not carry over to the test set as well as cross-validation suggested.

## Run

Download the competition data into `dataset/` (`train.csv`, `test.csv`, `gender_submission.csv`). On Kaggle the notebook finds the data under `/kaggle/input` automatically.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook titanic.ipynb
```

On macOS, XGBoost needs the OpenMP runtime (`brew install libomp`).
