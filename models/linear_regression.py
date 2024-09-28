import io
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
from qiskit_machine_learning.circuit.library import QNNCircuit
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

class Linear_regression:
    algorithm_globals.random_seed = 42

    # Train Regression
    def train_regression(__self__):
        num_samples = 20
        eps = 0.2
        lb, ub = -np.pi, np.pi
        X_ = np.linspace(lb, ub, num=50).reshape(50, 1)
        f = lambda x: np.sin(x)

        # Generate random training data
        X = (ub - lb) * algorithm_globals.random.random([num_samples, 1]) + lb
        y = f(X[:, 0]) + eps * (2 * algorithm_globals.random.random(num_samples) - 1)

        # Construct quantum circuits for feature map and ansatz
        param_x = Parameter("x")
        feature_map = QuantumCircuit(1, name="fm")
        feature_map.ry(param_x, 0)

        param_y = Parameter("y")
        ansatz = QuantumCircuit(1, name="vf")
        ansatz.ry(param_y, 0)

        # Combine feature map and ansatz into a QNN circuit
        qc = QNNCircuit(feature_map=feature_map, ansatz=ansatz)
        regression_estimator_qnn = EstimatorQNN(circuit=qc)

        # Define the regressor
        regressor = NeuralNetworkRegressor(
            neural_network=regression_estimator_qnn,
            loss="squared_error",
            optimizer=L_BFGS_B(maxiter=5),
        )

        # Train the regressor model
        regressor.fit(X, y)

        # Prepare final plot with predictions
        plt.plot(X_, f(X_), "r--", label="True function")  # Plot the true sine function
        plt.plot(X, y, "bo", label="Training data")  # Plot training data points

        # Predict the result using the trained regressor
        y_ = regressor.predict(X_)
        plt.plot(X_, y_, "g-", label="Model prediction")  # Plot predictions from the model

        # Add labels and legend to the plot
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Quantum Regression Result")
        plt.legend()

        # Save the plot to an image in memory
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return img
