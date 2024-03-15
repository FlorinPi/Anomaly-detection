import numpy as np


class DecisionNode:
    """Class to represent a single node in the decision tree."""

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

        # for leaf node
        self.value = value


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def _add_engineered_features(self, X):
        # Example: Add interaction terms or polynomial features
        feature_squared = X**2
        feature_01_product = X[:, 0] * X[:, 1]
        feature_sum = np.sum(X, axis=1)
        feature_mean = np.mean(X, axis=1)
        feature_std = np.std(X, axis=1)

        X_eng = np.column_stack((X, feature_squared, feature_01_product, feature_sum, feature_mean, feature_std))
        return X_eng

    def fit(self, X, y):
        X_eng = self._add_engineered_features(X)
        self.root = self._grow_tree(X_eng, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        # stopping criteria
        if depth >= self.max_depth or num_labels == 1:
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)

        # find best feature to split on
        feature_index, threshold = self._best_split_entropy(X, y, n_features)
        if feature_index is None:
            return DecisionNode(value=self._most_common_label(y))

        left_idxs, right_idxs = self._split(X[:, feature_index], threshold)
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return DecisionNode(feature_index=feature_index, threshold=threshold, left=left, right=right)


    def _best_split_entropy(self, X, y, n_features):
        best_feature, best_threshold = None, None
        best_entropy = float('inf')  # minimum possible value
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                entropy = self._entropy(X[:, feature_index], y, threshold)
                if entropy < best_entropy:
                    best_entropy = entropy
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold
    def _entropy(self, X_feature, y, threshold):
        # split dataset
        left_idxs, right_idxs = self._split(X_feature, threshold)
        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)
        if n_left == 0 or n_right == 0:
            return float('inf')

        # compute entropy for each child
        entropy_left = 0
        for label in np.unique(y[left_idxs]):
            p = np.sum(y[left_idxs] == label) / n_left
            entropy_left -= p * np.log2(p)

        entropy_right = 0
        for label in np.unique(y[right_idxs]):
            p = np.sum(y[right_idxs] == label) / n_right
            entropy_right -= p * np.log2(p)

        # weighted entropy
        return (n_left / n) * entropy_left + (n_right / n) * entropy_right

    def _split(self, X_feature, threshold):
        left_idxs = np.argwhere(X_feature <= threshold).flatten()
        right_idxs = np.argwhere(X_feature > threshold).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        X_eng = self._add_engineered_features(X)
        return np.array([self._traverse_tree(x, self.root) for x in X_eng])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

import pandas as pd

X_train = pd.read_csv("train.csv", usecols = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'])
y_train = pd.read_csv("train.csv", usecols = ['is_anomaly']).values.flatten().astype(int)

X_test = pd.read_csv("test.csv", usecols = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'])

# Convert X_test to a NumPy array

X_train = X_train.values
X_test = X_test.values

# Create a decision tree
clf = DecisionTree(max_depth=10)

X_train = clf._add_engineered_features(X_train)
X_test = clf._add_engineered_features(X_test)

clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

print(y_pred)

import csv

# Specify the filename for the CSV file
csv_filename = "dt.csv"

# Open the CSV file in write mode
with open(csv_filename, mode='w', newline='') as csv_file:
    # Create a CSV writer object
    csv_writer = csv.writer(csv_file)

    # Write each value in the array to the second column of the CSV file
    for value in y_pred:
        csv_writer.writerow([None, value])