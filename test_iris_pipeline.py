import matplotlib.pyplot as plt
from iris_pipeline import (
    load_dataset,
    train_model,
    evaluate_model,
    plot_feature,
    plot_features,
    plot_model,
)

def test_load_dataset():
    df = load_dataset()
    assert not df.empty, "The DataFrame should not be empty after loading the dataset."
    assert "species" in df.columns, "The dataset should contain a 'species' column."
def test_model_accuracy():
    df = load_dataset()
    model, X_train, X_test, y_train, y_test = train_model(df)
    acc = evaluate_model(model, X_test, y_test)
    assert acc > 0.8, f"Model accuracy is below 80%. Got {acc:.2f}"
def test_plot_feature_creates_figure():
    df = load_dataset()
    plt.close("all")  # reset figures before plotting
    plot_feature(df, "sepal length (cm)")
    figs = plt.get_fignums()
    assert len(figs) > 0, "plot_feature should create a figure."

def test_plot_features_creates_figure():
    df = load_dataset()
    plt.close("all")
    plot_features(df)
    figs = plt.get_fignums()
    assert len(figs) > 0, "plot_features should create a figure."

def test_plot_model_creates_figure():
    df = load_dataset()
    model, X_train, X_test, y_train, y_test = train_model(df)
    plt.close("all")
    plot_model(model, X_test, y_test)
    figs = plt.get_fignums()
    assert len(figs) > 0, "plot_model should create a figure."