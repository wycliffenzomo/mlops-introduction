import numpy as np
from Sklearn.datasets import load_iris
def load_dataset():
    iris_data = load_iris
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df['species'] = iris_data.target
    df["species_name"] = df.apply(
        lambda x: str(iris_data.target_names[int(x["species"])]), axis=1
    )
    return df

if __name__ == "__main__":
    iris_df = load_dataset()
    print(iris_df.head())

