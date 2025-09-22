import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
def test_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    return acc

if __name__ == "__main__": 
    iris_df = load_dataset()

    model, X_train, X_test, y_train, y_test = train_model(iris_df)

    acc = test_model(model, X_test, y_test)
    print(f"Accuracy: {acc:.2f}")

