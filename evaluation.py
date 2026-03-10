from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(model, X, y):

    print("Evaluating model...")

    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    return acc
