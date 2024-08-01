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

    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(hidden_layer_sizes=(50,), 
                        max_iter=10, verbose=10, random_state=1)
    MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(50,), learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=10,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=1, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=10,
              warm_start=False)
    mlp.fit(features_train, labels_train)
    print('Training score', mlp.score(features_train, labels_train))
    print('Testing score', mlp.score(features_test, labels_test))






if __name__ == "__main__":
    main()