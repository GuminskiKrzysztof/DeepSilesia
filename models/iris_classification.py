import io
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit_algorithms.utils import algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
import pandas as pd
import seaborn as sns
import time
import base64

class IrisQuantumClassifier:
    def __init__(self, random_seed=123, train_size=0.8):
        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.train_size = train_size
        self.features, self.labels = self.load_data()
        self.train_features, self.test_features, self.train_labels, self.test_labels = self.split_data()

        self.num_features = self.features.shape[1]
        self.feature_map = ZZFeatureMap(feature_dimension=self.num_features, reps=1)
        self.ansatz = RealAmplitudes(num_qubits=self.num_features, reps=3)
        self.optimizer = COBYLA(maxiter=100)
        self.vqc = VQC(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=self.optimizer,
        )

    def load_data(self):
        iris_data = load_iris()
        features = iris_data.data
        labels = iris_data.target
        return features, labels

    def split_data(self):
        train_features, test_features, train_labels, test_labels = train_test_split(
            self.features, self.labels, train_size=self.train_size, random_state=self.random_seed
        )
        return train_features, test_features, train_labels, test_labels

    def train_model(self):
        start = time.time()
        self.vqc.fit(self.train_features, self.train_labels)
        elapsed = time.time() - start
        return elapsed

    def evaluate_model(self):
        train_score = self.vqc.score(self.train_features, self.train_labels)
        test_score = self.vqc.score(self.test_features, self.test_labels)
        return train_score, test_score

    def plot_results(self, scores):
        plt.figure()
        plt.bar(['Train Score', 'Test Score'], scores, color=['blue', 'orange'])
        plt.ylim(0, 1)
        plt.title('Model Performance')
        plt.ylabel('Score')

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return img

    def execute_classification(self):
        elapsed_time = self.train_model()
        train_score, test_score = self.evaluate_model()
        scores = [train_score, test_score]
        img = self.plot_results(scores)

        return img, elapsed_time, train_score, test_score