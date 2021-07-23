from sklearn.feature_selection import SelectPercentile, chi2

def select_features(X_train, y_train, X_test):
    selector = SelectPercentile(chi2, percentile=15)
    selector.fit(X_train, y_train)

    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    return X_train, X_test
