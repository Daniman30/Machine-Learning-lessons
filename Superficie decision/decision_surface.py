import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from decision_tree_classifier import DecisionTreeClassifier
from random_forest_classifier import RandomForestClassifier
from distance_classifier import DistanceBasedClassifier
from knn_clasifier import KNearestNeighborsClassifier
from naive_bayes_classifier import NaiveBayesClassifier
from gradient_descent_classifier import GradientDescentLogisticRegression

def plot_decision_surface(X, y, model, title, test_size=0.3, random_state=42):
    """
    Grafica la superficie de decisión para un modelo de clasificación.

    Parámetros:
        X (numpy.ndarray): Datos de entrada (características).
        y (numpy.ndarray): Etiquetas de clase.
        model: Modelo de clasificación.
        title (str): Título del gráfico.
        test_size (float): Proporción del conjunto de prueba.
        random_state (int): Semilla para reproducibilidad.

    Devuelve:
        None: Muestra el gráfico de la superficie de decisión.
    """
    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Crear la malla para graficar la superficie de decisión
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Predecir las clases para cada punto de la malla
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Graficar la superficie de decisión
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

    # Graficar los puntos de entrenamiento y prueba
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', edgecolor='k', label='Entrenamiento', cmap=plt.cm.coolwarm)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', edgecolor='k', label='Prueba', cmap=plt.cm.coolwarm)

    plt.title(title)
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    plt.legend()
    plt.show()

# Generar datos de ejemplo
X, y = make_classification(
    n_samples=500, n_features=2, n_informative=2, n_redundant=0, 
    n_clusters_per_class=1, random_state=55
)

# plot_decision_surface(X, y, DistanceBasedClassifier(), "Distance")
# plot_decision_surface(X, y, DecisionTreeClassifier(max_depth=5), "Distance Tree")
# plot_decision_surface(X, y, RandomForestClassifier(n_estimators=10, max_depth=5), "Random Forest")
# plot_decision_surface(X, y, KNearestNeighborsClassifier(n_neighbors=5), "K Nearest Neighbors")
# plot_decision_surface(X, y, NaiveBayesClassifier(), "Naive Bayes")
# plot_decision_surface(X, y, GradientDescentLogisticRegression(lr=0.01, epochs=1000), "Naive Bayes")
