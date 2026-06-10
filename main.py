import numpy as np


def main():
    print("Logistic Regression With Neural Network")

def logistic_regression(sample_data: list, y: list):
    X = np.array(sample_data)
    y = np.array(y)
    n_features = X.shape[1]
    W = np.random.randn(n_features, 1)
    b = 0

    learning_rate = 0.01
    epochs = 1000



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

