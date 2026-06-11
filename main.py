import numpy as np


def main():
    print("Logistic Regression With Neural Network")

def initialize_parameter(dim):
    w = np.random.randn(dim, 1)
    b = 0

    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    cost = -1/m * np.sum(Y*np.log(A) + (1-Y) * np.log(1-A))

    dZ = A - Y
    dw = (1/m) * np.dot(X, dZ.T)
    db = (1/m) * np.sum(dZ)

    grads = {"dw": dw, "db": db}

    return grads, cost

def optimize(w, b, X, Y
             , learning_rate= 0.01,
             num_iterations= 1000
             ):
    costs = []
    grads = {}
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            print(f"Iteration {i}: cost = {cost}")

    params = {
        "w": w,
        "b": b,
    }

    return params, grads, costs

def predict(w, b, X):

    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    predictions =(A > 0.5).astype(int)
    return predictions

def model(X_train, Y_train, num_iterations=1000, learning_rate=0.01):
    n_features = X_train.shape[0]
    w, b = initialize_parameter(n_features)
    params, grads, costs = optimize(
        w,
        b,
        X_train, Y_train,
        learning_rate,
        num_iterations
    )
    w = params["w"]
    b = params["b"]

    Y_prediction = predict(w, b, X_train)

    accuracy = 100 - np.mean(np.abs(Y_prediction - Y_train))*100
    print(f"Accuracy:  {accuracy:.2f}%")

    return {
        "w": w,
        "b": b,
        "costs": costs,
        "accuracy": accuracy,
    }

def sigmoid(z):
    return 1/(1+np.exp(-z))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X_train = np.array([
        [2, 3, 5, 6],
        [1, 2, 4, 5]
    ])

    Y_train = np.array([
        [0, 0, 1, 1]
    ])

    result = model(
        X_train,
        Y_train,
        num_iterations=600,
        learning_rate=0.1
    )

    print("\nLearned Weights:")
    print(result["w"])

    print("\nLearned Bias:")
    print(result["b"])

