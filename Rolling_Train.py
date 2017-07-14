import pandas as pd
import sklearn.metrics as metrics
import numpy as np
import scipy
from math import sqrt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import PCA

NUM_CLASSES = 5
NUM_ROLLING = 20

def f(x):
    if x == 0 or x == 1:
        return 0
    elif x == 2:
        return 1
    elif x == 3:
        return 1
    elif x == 4:
        return 1
    else:
        return 2

def f1(x):
    if x == 0 or x == 1:
        return 0
    elif x == 2:
        return 1
    elif x == 3:
        return 2
    elif x == 4:
        return 3
    else:
        return 4


def load_and_partition():
    df = pd.read_excel('Final_Data_Part.xls', index_col='Date')
    label1 = df.label.apply(f).as_matrix().astype(int)
    label2 = df.label.apply(f).as_matrix().astype(int)
    data = df.iloc[ :, :df.shape[1] - 1]

    return data, label1, label2

def random_forest(x_train, y_train, n_features, num = 40):
    model = RandomForestClassifier(n_estimators = num, max_features = n_features)
    model.fit(x_train, y_train)
    return model

def gradient_boosting(x_train, y_train, num = 200):
    model = GradientBoostingClassifier(n_estimators=num)
    model.fit(x_train, y_train)
    return model

def Part_Norm_PC(i, j, k, n_components = 9):
    x_train = data.iloc[i:j,:]
    x_test = data.iloc[j:k, :]
    mean = x_train.mean(axis = 0)
    std = x_train.var(axis = 0).apply(lambda x: sqrt(x))
    std += 0.000001
    ma = x_train.max(axis = 0)
    mi = x_train.min(axis = 0)
    #x_train = (x_train - mean)/std
    #x_test = (x_test - mean)/std
    x_train = (x_train - mi)/(ma - mi + 0.0000000001)
    x_test = (x_test - mi)/(ma - mi + 0.000000001)
    pca = PCA(n_components=n_components)
    pca.fit(x_train)

    return pca.transform(x_train), label1[i: j], pca.transform(x_test), label1[j:k]

def train(n_features, num, i, n_components = 9):
    i = 150
    count = 0
    labels = []
    print('start training...')
    while(i < data.shape[0]):
        if (i <= data.shape[0] - NUM_ROLLING):
            x_train, labels_train, x_test, labels_test = Part_Norm_PC(i - 150, i, i+NUM_ROLLING, n_components)
        else:
            x_train, labels_train, x_test, labels_test = Part_Norm_PC(i - 150, i, data.shape[0], n_components)
        model = random_forest(x_train, labels_train, n_features = n_features, num = num)
        #model = gradient_boosting(x_train, labels_train, num = 300)
        pred_labels_train = model.predict(x_train)
        pred_labels_test = model.predict(x_test)
        labels += list(pred_labels_test)
        count += 1
        i += NUM_ROLLING
        #print("i = ", i)
        #print('the {0}th training is finished'.format(count))

    test = data.iloc[150:, :]
    labels_test = label1[150:]
    labels_test2 = label2[150:]
    pred_labels_test = np.array(labels)
    a = labels_test2[(pred_labels_test == 0)]
    b = labels_test2[(pred_labels_test == 2)]
    return sum(a == 0)/a.size - sum(a == 2)/a.size
'''
    print("showing results... when num = {0}, n_features = {1}".format(num, n_features))
    print("Train accuracy: {0}".format
        (metrics.accuracy_score(labels_train, pred_labels_train)))
#    print("Train accuracy on bull: {0}".format
#        (metrics.accuracy_score(labels_train[(labels_train == 0)],
#            pred_labels_train[(labels_train == 0)])))
#    print("Train accuracy on bear: {0}".format
#        (metrics.accuracy_score(labels_train[(labels_train == 4)],
#            pred_labels_train[(labels_train == 4)])))
    print("Test accuracy: {0}".format
        (metrics.accuracy_score(labels_test, pred_labels_test)))

    print("Test, predict bull when actually bull: {0}".format
        (sum(a == 0)/a.size))
    print("Test, predict bear when actually bear: {0}".format
        (sum(b == 2)/b.size))
    print("Test, predict bull when actually down: {0}".format
        (sum(a == 2)/a.size))
    print("Test, predict bear when actually up: {0}".format
        (sum(b == 0)/b.size))
'''



data, label1, label2 = load_and_partition()
d = {}
count = 0

for i in range(3, 16, 2):
    i_i = int(sqrt(i) + 1)
    for j in range(6, 13, 2):
        for k in range(100, 200, 20):
            d[(i, j, k)] = train(n_features = i_i, num = j, i = k, n_components = i)
            count += 1
            print('count = ', count)
