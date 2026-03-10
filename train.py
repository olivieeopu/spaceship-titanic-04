import pickle
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


RANDOM_STATE = 42


# 1 BASELINE MODEL
def baseline_lr(X, y):

    print("Training Logistic Regression Baseline...")

    model = LogisticRegression(random_state=RANDOM_STATE)

    cv_score = cross_val_score(
        model,
        X,
        y,
        cv=3,
        scoring="accuracy"
    ).mean()

    print(f"Baseline CV Accuracy: {cv_score:.4f}")

    return cv_score


# 2 OPTUNA OBJECTIVE
def objective_lr(trial, X, y):

    params = {
        "C": trial.suggest_float("C", 0.001, 100, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
        "max_iter": trial.suggest_int("max_iter", 100, 2000),
        "random_state": RANDOM_STATE
    }

    model = LogisticRegression(**params)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    score = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    ).mean()

    return score


# 3 OPTUNA TUNING
def tune_lr(X, y):

    print("Optimizing Logistic Regression...")

    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective_lr(trial, X, y),
        n_trials=30
    )

    print("Best Accuracy:", study.best_value)
    print("Best Params:", study.best_params)

    return study.best_params


# 4 TRAIN FINAL MODEL
def train_final_model(X, y, best_params):

    print("Training final Logistic Regression model...")

    model = LogisticRegression(**best_params)

    model.fit(X, y)

# save ke folder model
    with open("model/logistic_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained successfully!")
    print("Model saved to model/logistic_model.pkl")

    return model
