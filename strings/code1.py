import datetime
import io
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
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
from flask import Flask, redirect, url_for, render_template, send_file, jsonify
import base64
import time
from io import BytesIO 
import os

class Binary_classification:
    def __init__(self, num_samples=100, random_seed=42, num_inputs=4, num_qubits=4):
        self.random_seed = random_seed
        self.num_samples = num_samples
        self.num_inputs = num_inputs
        self.num_qubits = num_qubits
        print(num_samples)

    def generate_data(self):
        X = 2 * algorithm_globals.random.random((self.num_samples, self.num_inputs)) - 1
        y = 2 * (np.sum(X, axis=1) >= 0).astype(int) - 1
        return X, y

    def train_class(self, classifier, X, y):
        start_time = datetime.datetime.now()
        classifier.fit(X, y)
        end_time = datetime.datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        y_predict = classifier.predict(X)
        
        return execution_time, y_predict
    
    def plot_results(self, X, y, y_predict, title):
        for x, y_target, y_p in zip(X, y, y_predict):
            color = "bo" if y_target == 1 else "go"
            plt.plot(x[0], x[1], color)
            if y_target != y_p:
                plt.scatter(x[0], x[1], s=200, facecolors="none", edgecolors="r", linewidths=2)

        plt.plot([-1, 1], [1, -1], "--", color="black")
        plt.title(title)

        output_dir = "images"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "classification.png")
        plt.savefig(file_path, format='png')
        plt.close()
        return file_path 

    def train_classifier(self):
        X, y = self.generate_data()
        qc = QNNCircuit(num_qubits=self.num_qubits)
        estimator_qnn = EstimatorQNN(circuit=qc)
        classifier = NeuralNetworkClassifier(estimator_qnn, optimizer=COBYLA(maxiter=60))

        execution_time, y_predict = self.train_class(classifier, X, y)
        file_path = self.plot_results(X, y, y_predict, "Quantum Classification Result")

        return file_path, execution_time

if __name__ == "__main__":
    binary_classifier = Binary_classification()
    plot_path, exec_time = binary_classifier.train_classifier()
    print(f"Wykres zapisano pod ścieżką: {plot_path}")
    print(f"Czas wykonania: {exec_time} sekund")