import io
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_algorithms.utils import algorithm_globals
from sklearn.linear_model import LinearRegression
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit

class LinearRegressionModel:
    algorithm_globals.random_seed = 42

    def __init__(self, num_samples=50, random_seed=42, num_inputs=4):
        self.num_samples = num_samples
        self.num_inputs = num_inputs
        self.random_seed = random_seed
        self.eps = 0.2

    def generate_quantum_data(self):
        lb, ub = -np.pi, np.pi
        X = (ub - lb) * algorithm_globals.random.random([self.num_samples, 1]) + lb
        f = lambda x: np.sin(x)
        y = f(X[:, 0]) + self.eps * (2 * algorithm_globals.random.random(self.num_samples) - 1)
        return X, y

    def create_quantum_regressor(self):
        param_x = Parameter("x")
        feature_map = QuantumCircuit(1, name="fm")
        feature_map.ry(param_x, 0)

        param_y = Parameter("y")
        ansatz = QuantumCircuit(1, name="vf")
        ansatz.ry(param_y, 0)

        qc = QNNCircuit(feature_map=feature_map, ansatz=ansatz)
        regression_estimator_qnn = EstimatorQNN(circuit=qc)

        return NeuralNetworkRegressor(
            neural_network=regression_estimator_qnn,
            loss="squared_error",
            optimizer=L_BFGS_B(maxiter=5),
        )

    def train_quantum_regression(self):
        X, y = self.generate_quantum_data()
        regressor = self.create_quantum_regressor()

        start_time = datetime.datetime.now()
        regressor.fit(X, y)
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        return regressor, execution_time, X

    def plot_quantum_results(self, regressor, X, f):
        X_ = np.linspace(-np.pi, np.pi, 50).reshape(50, 1)
        plt.plot(X_, f(X_), "r--", label="True function")  # Just a red dashed line without markers
        # or
        plt.plot(X_, f(X_), "ro--", label="True function")  # Red dashed line with circular markers

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
        return img

    # def train_classical_regressor(self):
    #     X = np.linspace(-3, 3, self.num_samples).reshape(-1, 1)
    #     y = np.sin(X) + 0.1 * np.random.normal(size=X.shape)
    #     model = LinearRegression()

    #     start_time = datetime.datetime.now()
    #     model.fit(X, y)
    #     end_time = datetime.datetime.now()
    #     execution_time = (end_time - start_time).total_seconds()

    #     return model, execution_time, X, y

    # def plot_classical_results(self, model, X, y):
    #     X_pred = np.linspace(-3, 3, 100).reshape(-1, 1)
    #     y_pred = model.predict(X_pred)

    #     plt.plot(X, y, "bo", label="Training data")
    #     plt.plot(X_pred, y_pred, "g-", label="Model prediction")

    #     plt.xlabel("X")
    #     plt.ylabel("Y")
    #     plt.title("Classical Regression Result")
    #     plt.legend()

    #     img = io.BytesIO()
    #     plt.savefig(img, format='png')
    #     img.seek(0)
    #     plt.close()
    #     return img

    def execute_quantum_regression(self):
        regressor, execution_time, X = self.train_quantum_regression()
        img = self.plot_quantum_results(regressor, X, np.sin)
        return img, execution_time

    # def execute_classical_regression(self):
    #     model, execution_time, X, y = self.train_classical_regressor()
    #     img = self.plot_classical_results(model, X, y)
    #     return img, execution_time
