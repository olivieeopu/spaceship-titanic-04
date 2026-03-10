from data_ingestion import load_data
from preprocessing import feature_engineering, preprocess_data
from train import baseline_lr, tune_lr, train_final_model
from evaluation import evaluate_model


def run_pipeline():

    # 1 Load data
    df = load_data("data/train.csv")

    # 2 Feature engineering
    df = feature_engineering(df)

    # 3 Preprocessing
    X, y, feature_columns = preprocess_data(df)

    # 4 Baseline model
    baseline_lr(X, y)

    # 5 Hyperparameter tuning
    best_params = tune_lr(X, y)

    # 6 Train final model
    model = train_final_model(X, y, best_params)

    # 7 Evaluation
    evaluate_model(model, X, y)


if __name__ == "__main__":
    run_pipeline()
