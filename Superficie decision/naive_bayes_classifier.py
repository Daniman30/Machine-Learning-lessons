import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}
    
    def fit(self, X, y):
        """
        Entrena el modelo Naive Bayes con los datos de entrada X y etiquetas y.
        """
        # Identificar las clases únicas
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        # Calcular la media, varianza y probabilidad a priori para cada clase
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.priors[c] = X_c.shape[0] / n_samples
    
    def _gaussian_density(self, class_idx, x):
        """
        Calcula la densidad de probabilidad de una distribución normal (Gaussian).
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    def predict(self, X):
        """
        Predice las clases para los datos de entrada X.
        """
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        """
        Predice la clase para una sola muestra.
        """
        posteriors = []
        
        for c in self.classes:
            # Inicializar con la probabilidad a priori
            prior = np.log(self.priors[c])
            # Sumar las probabilidades logarítmicas de las características
            conditional = np.sum(np.log(self._gaussian_density(c, x)))
            posterior = prior + conditional
            posteriors.append(posterior)
        
        # Retorna la clase con la mayor probabilidad posterior
        return self.classes[np.argmax(posteriors)]
    