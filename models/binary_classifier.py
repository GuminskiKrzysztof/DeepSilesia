import datetime
import io
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification
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

class Binary_classifier:
    algorithm_globals.random_seed = 42
    def train_classifier(__self__):
        num_inputs = 2
        num_samples = 20
        X = 2 * algorithm_globals.random.random([num_samples, num_inputs]) - 1
        y01 = 1 * (np.sum(X, axis=1) >= 0)  # in { 0,  1}
        y = 2 * y01 - 1  # in {-1, +1}

        # Construct QNN with the QNNCircuit's default ZZFeatureMap and RealAmplitudes ansatz
        qc = QNNCircuit(num_qubits=2)
        estimator_qnn = EstimatorQNN(circuit=qc)

        # Create the classifier
        classifier = NeuralNetworkClassifier(
            estimator_qnn, optimizer=COBYLA(maxiter=60)
        )

        # Train the classifier
        classifier.fit(X, y)

        # Predict the data points
        y_predict = classifier.predict(X)

        # Prepare plot for results
        for x, y_target, y_p in zip(X, y, y_predict):
            if y_target == 1:
                plt.plot(x[0], x[1], "bo")
            else:
                plt.plot(x[0], x[1], "go")
            if y_target != y_p:
                plt.scatter(x[0], x[1], s=200, facecolors="none", edgecolors="r", linewidths=2)
        plt.plot([-1, 1], [1, -1], "--", color="black")
        plt.title("Quantum Classification Result")
        
        # Save the plot to a PNG image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return img

    

    def train_classical_classifier(__self__):
        num_inputs = 2
        num_samples = 20
        X = 2 * np.random.random([num_samples, num_inputs]) - 1
        y01 = 1 * (np.sum(X, axis=1) >= 0)  # in { 0,  1}
        y = 2 * y01 - 1  # in {-1, +1}

        # Create the logistic regression classifier
        classifier = LogisticRegression()

        # Train the classifier
        date1 = datetime.datetime.now()
        classifier.fit(X, y)
        date2 = datetime.datetime.now()
        diff1 = date2 - date1
        print(f"Training time: {diff1}")

        # Predict the data points
        y_predict = classifier.predict(X)

        # Prepare plot for results
        for x, y_target, y_p in zip(X, y, y_predict):
            if y_target == 1:
                plt.plot(x[0], x[1], "bo")  # True positive
            else:
                plt.plot(x[0], x[1], "go")  # True negative
            if y_target != y_p:
                plt.scatter(x[0], x[1], s=200, facecolors="none", edgecolors="r", linewidths=2)  # Misclassified points

        plt.plot([-1, 1], [1, -1], "--", color="black")  # Decision boundary
        plt.title("Classical Classification Result")

        # Save the plot to a PNG image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return img