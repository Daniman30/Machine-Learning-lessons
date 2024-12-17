from sklearn import neighbors
import random
import math
from sklearn.datasets import make_classification

# Stage 1
# Select K neighbors number

# Stage 2
# Calculate distance

# Stage 3
# Take K nearest neighbors based on calculated distance

# Stage 4
# Among the K neighbors count the number of points in each category

# Stage 5
# Assign a new point to the most present category among the K neighbors 

# Stage 6 
# Model is ready.

def train_test_split(X, y, test_size=0.3):
    """
    Splits the dataset into training and test sets.

    Parameters:
    - X: List of features (list of lists or list of arrays)
    - y: List of labels (list)
    - test_size: Proportion of the test set (e.g. 0.3 for 30%)

    Returns:
    - X_train: Feature set for training
    - X_test: Feature set for testing
    - y_train: Labels for training
    - y_test: Labels for testing
    """

    # Mix data and labels together to maintain correspondence
    data = list(zip(X, y))
    random.shuffle(data)
    
    # Calculate the amount of data in the test set
    test_count = int(len(X) * test_size)
    
    # Split the data into training and test sets
    test_data = data[:test_count]
    train_data = data[test_count:]
    
    # Separate features and labels for training and testing
    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)
    
    # Convert to lists
    return list(X_train), list(X_test), list(y_train), list(y_test)

def calculate_distance_euclidienne(v1, v2):
    sum = 0
    for i in range(len(v1)):
        sum += abs(v1[i] - v2[i])
    return sum

def calculate_distance_manhattan(v1, v2):
    sum = 0
    for i in range(len(v1)):
        sum += (v1[i] - v2[i]) ** 2
    return math.sqrt(sum)

def find_neighbors(X_train, y_train, new_vector, k):
    neighbors = []
    for i in range(len(X_train)):
        distance = calculate_distance_euclidienne(X_train[i], new_vector)
        neighbors.append((distance, y_train[i]))
    
    neighbors.sort(key=lambda x: x[0])
    final_neighbors = [distance[1] for distance in neighbors[:k]]
    return final_neighbors

def predict(X_train, y_train, new_vector, k):
    neighbors = find_neighbors(X_train, y_train, new_vector, k)
    categories = {}
    for n in neighbors:
        categories[n] = categories.get(n, 0) + 1
    ordered = dict(sorted(((k, v) for k, v in categories.items()), key=lambda x: x[1], reverse=True))
    print(categories)
    return list(ordered.keys())[0]

def calculate_precision(y_true, y_pred):
    hits = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return hits / len(y_true)

# X = [[1, 2], [2, 3], [3, 3], [6, 5], [7, 8], [8, 8]]
# y = ["atun", "atun", "atun", "salmon", "salmon", "salmon"]
# k = 3
# nuevos_datos = [5, 5]
X, y = make_classification(
    n_samples=500, n_features=2, n_informative=2, n_redundant=0, 
    n_clusters_per_class=1, random_state=55
)
k = 5
print(y[0])
nuevos_datos = [1, 1]

x = predict(X, y, nuevos_datos, k)
print(x)