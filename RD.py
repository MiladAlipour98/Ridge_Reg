import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def load_data(path, delimiter = ","):
    df = pd.read_csv(path, header=None)
    return df.to_numpy()

def append_bias(X):
    return np.append(X, np.ones((X.shape[0], 1)), axis=1)


def hypothesis(w, X):
    return X @ w


def initialize_with_zeros(n_features):
    return np.zeros((n_features + 1, 1))



def calc_cost(w, X, Y):
    X = append_bias(X)
    n = X.shape[0]
    Y = Y.reshape(n,1)

    A = hypothesis(w, X)
    cost = (1 / (2 * n)) * np.sum((A - Y) ** 2)

    return cost

def calc_beta(w, X, Y):
    X = append_bias(X)
    n = X.shape[0]
    Y = Y.reshape(n,1)

    beta =np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    return beta

def calc_dw(w, X, Y):
    X = append_bias(X)
    n = X.shape[0]
    Y = Y.reshape(n, 1)

    A = hypothesis(w, X)
    dw = (1 / n) * (X.T @ (A - Y))
    return dw

def btls_approach(w, X, Y, alpha, beta):
    t = 1
    while (calc_cost(w - (t * calc_dw(w, X, Y)), X, Y)) > (calc_cost(w, X, Y) - (alpha * t * np.linalg.norm(calc_dw(w, X, Y)) ** 2)):
        t *= beta
    return t


def gradient_descent(w, X, Y, iters):
    alpha = 0.5
    beta = 0.5
    for i in range(iters):
        cost = calc_cost(w, X, Y)
        dw = calc_dw(w, X, Y)
        t = btls_approach(w,X,Y,alpha,beta)
        w -= t * dw


    return w, cost

def model_Ridge(w, X, Y, iters,penalty,learning_rate):
    X = append_bias(X)
    n = X.shape[0]
    b = 0

    f_k = np.zeros(iters)
    for i in range(iters):
        Y_pred = np.reshape((X.dot(w) + b), (-1,1))
        db = -2 * np.sum(Y - Y_pred) / n
        dw = (- (2 * (X.T).dot(Y - Y_pred)) + (2 * penalty * w)) / n
        w = w - learning_rate * dw
        b = b - learning_rate * db
        f_k[i] = np.sum((X @ w) + b) / n

    return w, b ,f_k


def main():

    xs = load_data("X_train.csv")
    ys = load_data("Y_train.csv")
    xs = normalize(xs)
    n = xs.shape[1]

    w = initialize_with_zeros(n)
    # 1
    start_time = time.time()
    cost = calc_cost(w, xs, ys)
    beta = calc_beta(w, xs, ys)
    print("Beta values are =", beta)
    print("Cost of function is=", cost)
    print("--- %s seconds ---" % (time.time() - start_time))


    start_time = time.time()
    betaa, costs = gradient_descent(w, xs, ys, 10000)
    print("Beta values are =", betaa)
    print("Cost of function is=", costs)
    print("--- %s seconds ---" % (time.time() - start_time))

    # 2
    f_s = 57.0410
    ws, b, f_k = model_Ridge(w, xs, ys, 500, 1, 0.00001)
    print("Beta0 value is =", b)
    print("Beta values are =", ws)
    print("Convergencevalue for step size = 0.001 is :", f_s - f_k[-1])

    # plot
    K = list(range(1, 501))
    logf = f_s - f_k
    plt.plot(figsize=(10, 6))
    plt.semilogy(K, logf, color='red')
    plt.title('Semilogy Plot for step size = 0.00001')
    plt.xlabel('$K$')
    plt.ylabel('$f* - fk$')
    plt.show()



if __name__ == '__main__':
    main()




