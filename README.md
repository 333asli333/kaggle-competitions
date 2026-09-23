# Kaggle Competitions

My notebooks for [Kaggle](https://www.kaggle.com/competitions) competitions. Each competition lives in its own folder with an English notebook, a Turkish translation and a short summary.

| Competition | Task | Approach | CV accuracy | Kaggle public score |
|---|---|---|---|---|
| [Titanic](titanic/) | Binary classification | Logistic Regression, nested CV, statistical model comparison | 83.1% | 0.76794 |
| [Spaceship Titanic](spaceship-titanic/) | Binary classification | Rule-based imputation, leak-free pipeline, Optuna tuning, ensemble of four gradient boosting models | 81.5% | 0.81038 |

## Principles

- **Honest evaluation:** leak-free pipelines, nested cross-validation for tuning, statistical tests before claiming one model is better.
- **Data first:** every claim in a notebook is shown by the data or backed by a source link.
- **Readable:** notebooks are written as a story, with technical details kept in separate notes.

## Running locally

Each folder has its own `requirements.txt`. Competition data is **not** included in this repository; download it from the competition page into the folder's `dataset/` directory.

```bash
cd titanic
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
