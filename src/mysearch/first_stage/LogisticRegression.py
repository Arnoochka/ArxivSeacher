import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model._base import LinearClassifierMixin
from scipy.sparse import csr_matrix
from numpy.typing import NDArray
from mysearch.utils import Parallelizer

class LogisticRegression(BaseEstimator, LinearClassifierMixin):
    def __init__(self,
                 learning_rate=0.01,
                 max_iterations=1000,
                 tolerance=1e-6, 
                 C=1.0):
        
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.C = C
        self.classes_ = None
        self.weights_ = None
        self.bias_ = None

    def _sigmoid(self, z: NDArray) -> NDArray:
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, X: csr_matrix, y: NDArray, weights: NDArray, bias: float) -> float:
        n_samples = X.shape[0]
        p = self._sigmoid(X.dot(weights) + bias)
        # Log LH
        cost = (-1/n_samples) * (y @ np.log(p + 1e-15) + (1 - y) @ np.log(1 - p + 1e-15))
        # L2 
        reg_term = (1/(2 * self.C * n_samples)) * np.sum(weights**2)
        cost += reg_term
            
        return cost
    
    def _compute_gradient(self,
                          X: csr_matrix,
                          y: NDArray,
                          weights: NDArray,
                          bias: float) -> tuple[NDArray, float]:
        n_samples = X.shape[0]
        err = self._sigmoid(X.dot(weights) + bias) - y
        grad_weights = (1/n_samples) * X.T.dot(err)
        grad_bias = (1/n_samples) * np.sum(err)
        
        # L2
        grad_weights += weights / (self.C * n_samples)
            
        return grad_weights, grad_bias
    
    def _fit_binary(self, X: csr_matrix, y: NDArray) -> tuple[NDArray, float]:
        n_features = X.shape[1]
        weights = np.zeros(n_features, dtype=np.float64)
        bias = 0.0
        
        prev_cost = float('inf')
        for _ in range(self.max_iterations):
            grad_w, grad_b = self._compute_gradient(X, y, weights, bias)
            weights -= self.learning_rate * grad_w
            bias -= self.learning_rate * grad_b
            cost = self._compute_cost(X, y, weights, bias)
            if abs(prev_cost - cost) < self.tolerance:
                break
            prev_cost = cost
            
        return weights, bias
    
    def fit(self, X: csr_matrix, y: NDArray) -> "LogisticRegression":
        y = np.asarray(y, dtype=np.int32)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        n_features = X.shape[1]
        self.weights_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.bias_ = np.zeros(n_classes, dtype=np.float64)
        
        for i, class_label in enumerate(self.classes_):
            binary_y = (y == class_label).astype(np.float64)
            weights, bias = self._fit_binary(X, binary_y)
            self.weights_[i] = weights
            self.bias_[i] = bias
            
        return self
    
    def predict_proba(self, X: csr_matrix) -> NDArray:
        """
        P(y = k | x) = p_k / sum(p_k), где p_k = sigmoid(X @ w_k + b_k)
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_samples, n_classes), dtype=np.float64)
        
        for с in range(n_classes):
            z = X.dot(self.weights_[с]) + self.bias_[с]
            probabilities[:, с] = self._sigmoid(z)
        row_sums = probabilities.sum(axis=1, keepdims=True)
        probabilities /= row_sums
        
        return probabilities
    
    def predict(self, X: csr_matrix, k: int = 3) -> NDArray:
        probabilities = self.predict_proba(X)
        predictions = np.argsort(-probabilities, axis=1)[:, :k]
        return self.classes_[predictions]
    
    