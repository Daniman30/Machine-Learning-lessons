import numpy as np

def gradient_descent(X, y, lr=0.01, epochs=1000):
    """
    Implementa el Descenso por Gradiente para una regresión lineal simple.
    
    Parámetros:
    - X: Matriz de características (n_samples, n_features)
    - y: Vector de etiquetas (n_samples,)
    - lr: Tasa de aprendizaje (learning rate)
    - epochs: Número de iteraciones
    
    Retorna:
    - w: Pesos ajustados
    - b: Sesgo ajustado
    - history: Historial de pérdida en cada iteración
    """
    n_samples, n_features = X.shape
    
    # Inicializar pesos y sesgo
    w = np.zeros(n_features)  # Pesos (vector de coeficientes)
    b = 0                     # Sesgo (intercepto)
    
    # Historial de pérdida
    history = []
    
    for epoch in range(epochs):
        # Predicciones
        y_pred = np.dot(X, w) + b
        
        # Cálculo del error (residuales)
        error = y_pred - y
        
        # Gradientes
        dw = (1 / n_samples) * np.dot(X.T, error)  # Derivada respecto a w
        db = (1 / n_samples) * np.sum(error)      # Derivada respecto a b
        
        # Actualización de los parámetros
        w -= lr * dw
        b -= lr * db
        
        # Calcular y guardar la pérdida (MSE: Mean Squared Error)
        loss = (1 / (2 * n_samples)) * np.sum(error ** 2)
        history.append(loss)
        
        # Imprimir el progreso cada 100 iteraciones
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return w, b, history

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de ejemplo (regresión lineal simple)
    X = np.array([[1], [2], [3], [4], [5]])  # Característica (1D)
    y = np.array([3, 6, 9, 12, 15])          # Etiquetas (y = 3x)
    
    # Ajustar el modelo
    w, b, history = gradient_descent(X, y, lr=0.01, epochs=1000)
    
    # Resultados
    print("\nPesos finales:", w)
    print("Sesgo final:", b)
