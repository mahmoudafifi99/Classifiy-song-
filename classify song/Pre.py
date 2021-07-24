import numpy as np
from sklearn.model_selection import train_test_split




def featureScaling(values, a, b):
    Normalized_values = np.zeros((values.shape[0]))
    Normalized_values = ((values - min(values)) / (max(values) - min(values))) * (b - a) + a
    return Normalized_values

def split(data,ydata):
    X_train, X_test, y_train, y_test = train_test_split(data, ydata, test_size=0.2, random_state=0, shuffle=True)
    tested(X_test,y_test)
    return X_train, X_test, y_train, y_test


def tested(Xtest,ytest):

    return Xtest,ytest


