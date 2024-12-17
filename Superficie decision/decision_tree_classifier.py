import numpy as np
from collections import Counter

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        def build_tree(X, y, depth):
            if len(set(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
                return Counter(y).most_common(1)[0][0]

            best_feature = np.argmax([np.var(X[:, i]) for i in range(X.shape[1])])
            threshold = np.median(X[:, best_feature])

            left_mask = X[:, best_feature] <= threshold
            right_mask = ~left_mask

            return {
                'feature': best_feature,
                'threshold': threshold,
                'left': build_tree(X[left_mask], y[left_mask], depth + 1),
                'right': build_tree(X[right_mask], y[right_mask], depth + 1),
            }

        self.tree = build_tree(X, y, 0)

    def predict(self, X):
        def traverse_tree(tree, x):
            if not isinstance(tree, dict):
                return tree
            if x[tree['feature']] <= tree['threshold']:
                return traverse_tree(tree['left'], x)
            else:
                return traverse_tree(tree['right'], x)

        return np.array([traverse_tree(self.tree, x) for x in X])
    
# Entropy equation
# Entropy(p) = - sum(pi * log(2, pi)) for pi in probabilities
# https://www.analyticsvidhya.com/blog/2020/11/entropy-a-key-concept-for-all-data-science-beginners/
