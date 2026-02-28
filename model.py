import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

def train_model():
    dataset = pd.read_csv("data/Investment.csv")

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    X = pd.get_dummies(X, drop_first=True)

    model = LinearRegression()
    model.fit(X, y)

    pickle.dump(model, open("model.pkl", "wb"))

if __name__ == "__main__":
    train_model()
