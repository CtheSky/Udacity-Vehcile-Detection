import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import load_dataset
import feature_extract

filename = 'classifier'


def train_classifier():
    features = feature_extract.get_features()
    labels = load_dataset.get_labels()

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=7)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC())
    ])

    clf.fit(X_train, y_train)
    joblib.dump(clf, filename)

    print('Validation accuracy:', clf.score(X_test, y_test))

    return clf


def get_classifier(use_cache=True):
    if use_cache and os.path.exists(filename):
        return joblib.load(filename)
    else:
        return train_classifier()


if __name__ == '__main__':
    train_classifier()
