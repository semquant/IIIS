import pandas as pd
import sklearn.metrics as metrics
import numpy as np
import scipy
dim = 90
d = 400
w = np.random.normal(0, 0.1, (dim, d))
b = np.random.uniform(0, 2*np.pi, d)

NUM_CLASSES = 3

def f(x):
    if x == 0 or x == 1:
        return 0
    elif x >= 2 and x <= 4:
        return 1
    elif x > 4:
        return 2

def load_and_partition(random = True):
    '''load data with randomly partition'''
    df = pd.read_excel('Final_Data_Part.xls', index_col='Date')
    df.label = df.label.apply(f)
    if random:
        data = df.as_matrix()
        arr = np.arange(data.shape[0])
        np.random.seed(0)
        np.random.shuffle(arr)
        arr = (arr > data.shape[0]*0.8)
        train = data[~arr,:]
        test = data[arr,:]
    else:
        train = df[df.index > '2007-06-01'].as_matrix()
        test = df[df.index <= '2007-06-01'].as_matrix()
    x_train = train[:,:train.shape[1]-1]
    y_train = train[:,train.shape[1]-1].astype(int)
    x_test = test[:,:test.shape[1]-1]
    y_test = test[:,test.shape[1]-1].astype(int)
    return (x_train, y_train), (x_test, y_test)

def train(X_train, y_train, reg=0):
    ''' Build a model from X_train -> y_train '''
    xtx = X_train.T.dot(X_train)
    return scipy.linalg.solve(xtx + reg*np.eye(xtx.shape[0]),
        X_train.T.dot(y_train), sym_pos=True)

def one_hot(labels):
    return np.eye(NUM_CLASSES)[labels].astype(int)

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    return np.argmax(X.dot(model), axis=1)

def phi(X):
    ''' Multiply the 90-dimensional vectors by unit normal '''
    return np.cos(X.dot(w) + b)

if __name__ == "__main__":
    (x_train, labels_train), (x_test, labels_test) = load_and_partition(random = False)
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)
    x_train, x_test = phi(x_train), phi(x_test)

    model = train(x_train, y_train, reg=0.03)
    pred_labels_train = predict(model, x_train)
    pred_labels_test = predict(model, x_test)
    a = pred_labels_test[(labels_test == 2)]
    b = pred_labels_test[(labels_test == 0)]
    print("linear regression algorithm")
    print("Train accuracy: {0}".format
        (metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Train accuracy on bull: {0}".format
        (metrics.accuracy_score(labels_train[(labels_train == 0)],
            pred_labels_train[(labels_train == 0)])))
    print("Train accuracy on bear: {0}".format
        (metrics.accuracy_score(labels_train[(labels_train == 2)],
            pred_labels_train[(labels_train == 2)])))
    print("Test accuracy: {0}".format
        (metrics.accuracy_score(labels_test, pred_labels_test)))
    print("Test accuracy on bull: {0}".format
        (metrics.accuracy_score(labels_test[(labels_test == 0)],
            pred_labels_test[(labels_test == 0)])))
    print("Test accuracy on bear: {0}".format
        (metrics.accuracy_score(labels_test[(labels_test == 2)],
            pred_labels_test[(labels_test == 2)])))
    print("Test, predict bull when actually bear: {0}".format
        (sum(a == 0)/a.size))
    print("Test, predict bear when actually bull: {0}".format
        (sum(b == 2)/b.size))
    print("------------------------------")
