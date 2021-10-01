import pickle
from sklearn.feature_selection import SelectPercentile, chi2

def select_features(X_train, y_train, X_test, is_predict = False):
    if is_predict:
        selector = pickle.load(open('feature_selector.sav', 'rb'))
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)           
    else:
        selector = SelectPercentile(chi2, percentile=15)
        selector.fit(X_train, y_train)

        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)

        pickle.dump(selector, open('feature_selector.sav', 'wb'))

    return X_train, X_test
