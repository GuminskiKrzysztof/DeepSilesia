import io
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit_algorithms.utils import algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from sklearn.linear_model import LinearRegression
from qiskit_machine_learning.circuit.library import QNNCircuit
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import time

class Linear_regression:
    algorithm_globals.random_seed = 42

    def __init__(self, num_samples=50, random_seed=42, num_inputs=4):
        self.num_samples = num_samples
        self.num_inputs = num_inputs
        self.random_seed = random_seed
    

    def train_regression(self):
        num_samples = self.num_samples
        eps = 0.2
        lb, ub = -np.pi, np.pi
        X_ = np.linspace(lb, ub, num=num_samples).reshape(50, 1)
        f = lambda x: np.sin(x)

        X = (ub - lb) * algorithm_globals.random.random([num_samples, 1]) + lb
        y = f(X[:, 0]) + eps * (2 * algorithm_globals.random.random(num_samples) - 1)

        param_x = Parameter("x")
        feature_map = QuantumCircuit(1, name="fm")
        feature_map.ry(param_x, 0)

        param_y = Parameter("y")
        ansatz = QuantumCircuit(1, name="vf")
        ansatz.ry(param_y, 0)

        qc = QNNCircuit(feature_map=feature_map, ansatz=ansatz)
        regression_estimator_qnn = EstimatorQNN(circuit=qc)

        regressor = NeuralNetworkRegressor(
            neural_network=regression_estimator_qnn,
            loss="squared_error",
            optimizer=L_BFGS_B(maxiter=5),
        )

        start_time = datetime.datetime.now()
        regressor.fit(X, y)
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        plt.plot(X_, f(X_), "ra--", label="True function")
        plt.plot(X, y, "bo", label="Training data")

        y_ = regressor.predict(X_)
        plt.plot(X_, y_, "g-", label="Model prediction")

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Quantum Regression Result")
        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return img, execution_time


    def train_classical_regressor(self):
        num_samples = self.num_samples
        X = np.linspace(-3, 3, num_samples).reshape(-1, 1)
        y = np.sin(X) + 0.1 * np.random.normal(size=X.shape)

        model = LinearRegression()
        start_time = datetime.datetime.now()
        model.fit(X, y)
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        X_pred = np.linspace(-3, 3, 100).reshape(-1, 1)
        y_pred = model.predict(X_pred)

        plt.plot(X, y, "bo", label="Dane treningowe")
        plt.plot(X_pred, y_pred, "g-", label="Predykcja modelu")

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Wynik regresji klasycznej (Regresja Liniowa)")
        plt.legend()

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return img, execution_time
