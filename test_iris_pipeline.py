from iris_pipeline import load_dataset, train_model, evaluate_model

def test_load_dataset():
    df = load_dataset()
    assert not df.empty, "The DataFrame should not be empty after loading the dataset."

def test_model_accuracy():
    df = load_dataset()
    model, X_train, X_test, y_train, y_test = train_model(df)
    acc = evaluate_model(model, X_test, y_test)
    assert acc > 0.8, f"Model accuracy is below 80%. Got {acc:.2f}"