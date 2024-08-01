import pandas as pd
import numpy as np
import math

import imageio
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor

def main():
    df = pd.read_csv('data.csv')
    decision_tree(df)




def decision_tree(df):
    features = df[['Year', 'Position', 'Value_cap_space', 'Value_draft_data', 'Previous_AV']]
    features = pd.get_dummies(features)
    labels = df['Current_AV']
    features_train, features_test = features[:864], features[864:]
    labels_train, labels_test = labels[:864], labels[864:]
    model = DecisionTreeRegressor()

    # Train it on the **training set**
    model.fit(features_train, labels_train)

    # Compute training accuracy
    train_predictions = model.predict(features_train)
    print('Tree Train Accuracy:', accuracy_score(labels_train, train_predictions))

    # Compute test accuracy
    test_predictions = model.predict(features_test)
    print('Tree Test  Accuracy:', accuracy_score(labels_test, test_predictions))


    learning_rates = [0.001, 0.01, 0.5]
    sizes = [(10,), (50,), (10, 10, 10, 10),]
    for learning_rate in learning_rates:
        for size in sizes:
            print(f'Learning Rate {learning_rate}, Size {size}')
            mlp = MLPRegressor(hidden_layer_sizes=size, max_iter=10,
                                random_state=1, learning_rate_init=learning_rate)
            mlp.fit(features_train, labels_train)
            print("    Training set score: %f" % mlp.score(features_train, labels_train))
            print("    Test set score: %f" % mlp.score(features_test, labels_test))





if __name__ == "__main__":
    main()