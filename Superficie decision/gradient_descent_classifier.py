import numpy as np

class GradientDescentLogisticRegression:
    """
    Implementación de Regresión Logística usando Descenso por Gradiente.
    """
    def __init__(self, lr=0.01, epochs=1000):
        """
        Inicializa los parámetros del modelo.
        
        Parámetros:
        - lr: Tasa de aprendizaje (learning rate).
        - epochs: Número de iteraciones del descenso por gradiente.
        """
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
        self.history = []

    def sigmoid(self, z):
        """
        Función sigmoide para calcular probabilidades.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Ajusta el modelo a los datos usando Descenso por Gradiente.
        
        Parámetros:
        - X: Matriz de características (n_samples, n_features).
        - y: Vector de etiquetas (n_samples,).
        """
        n_samples, n_features = X.shape
        # Inicializar pesos y sesgo
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.epochs):
            # Predicciones (probabilidades)
            linear_model = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(linear_model)

            # Gradientes
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Actualización de parámetros
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Calcular pérdida (log loss)
            loss = -(1 / n_samples) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            self.history.append(loss)

            # Imprimir progreso cada 100 iteraciones
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Realiza predicciones usando el modelo ajustado.
        
        Parámetros:
        - X: Matriz de características (n_samples, n_features).
        
        Retorna:
        - y_pred_class: Vector de etiquetas predichas (n_samples,).
        """
        linear_model = np.dot(X, self.w) + self.b
        y_pred = self.sigmoid(linear_model)
        # Convertir probabilidades a etiquetas binarias
        y_pred_class = [1 if prob >= 0.5 else 0 for prob in y_pred]
        return np.array(y_pred_class)
