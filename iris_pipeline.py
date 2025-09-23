import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def load_dataset():
    iris_data = load_iris()
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df['species'] = iris_data.target
    df["species_name"] = df.apply(
        lambda x: str(iris_data.target_names[int(x["species"])]), axis=1
    )
    return df

if __name__ == "__main__":
    iris_df = load_dataset()
    print(iris_df.head())

# Function to train the model
def train_model(data):
    X = data.iloc[:, :-2]  
    y = data["species"]    

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Creating and and training a Logistic Regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

# Function for testing the model's performance
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    return acc

if __name__ == "__main__": 
    iris_df = load_dataset()

    model, X_train, X_test, y_train, y_test = train_model(iris_df)

    acc = test_model(model, X_test, y_test)
    print(f"Accuracy: {acc:.2f}")

def plot_feature(df, column_name):
    plt.figure()  
    plt.hist(df[column_name])
    plt.title(column_name)


def plot_features(df):
    plt.figure() 
    for col in df.columns:
        plt.hist(df[col], alpha=0.5, label=col)
    plt.legend()
    plt.title("Feature Histograms")

def plot_model(model, X_test, y_test):
    plt.figure()  # <- ensure a figure exists
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")


if __name__ == "__main__":
    iris_df = load_dataset()
    model, X_train, X_test, y_train, y_test = train_model(iris_df)
    acc = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {acc:.2f}")

    plot_feature(iris_df, "sepal length (cm)")
    plot_features(iris_df)
    plot_model(model, X_test, y_test)