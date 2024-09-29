document.addEventListener("DOMContentLoaded", function () {
    let selectedMode = null;

    // Event listeners for buttons
    document.getElementById("classification-button").addEventListener("click", function () {
        editor.setValue(
            `import datetime
import io
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return img

    def train_classifier(self):
        X, y = self.generate_data()
        qc = QNNCircuit(num_qubits=self.num_qubits)
        estimator_qnn = EstimatorQNN(circuit=qc)
        classifier = NeuralNetworkClassifier(estimator_qnn, optimizer=COBYLA(maxiter=60))

        execution_time, y_predict = self.train_class(classifier, X, y)
        img = self.plot_results(X, y, y_predict, "Quantum Classification Result")

        return img, execution_time`
        )


        document.getElementById('code-editor').textContent = "Code editor (classification)";
        selectedMode = "train_quantum";    
        
    });

    document.getElementById("regression-button").addEventListener("click", function () {
        editor.setValue(
            `import io
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

    def execute_quantum_regression(self):
        regressor, execution_time, X = self.train_quantum_regression()
        img = self.plot_quantum_results(regressor, X, np.sin)
        return img, execution_time
`);

        document.getElementById('code-editor').textContent = "Code editor (regression)";
        selectedMode = "regress";
    });

    document.getElementById("iris-button").addEventListener("click", function () {
        editor.setValue(
            `import io
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
        plt.bar(['Train Score', 'Test Score'], scores, color=['#ff79bd', '#eeff73'])
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

        return img, elapsed_time, train_score, test_score`
        )

        document.getElementById('code-editor').textContent = "Code editor (iris classification)";
        selectedMode = "iris";        
        
    });

    // Handle "Run" button
    document.getElementById("run-button").addEventListener("click", function () {
        document.getElementById('run-info').textContent = "Running...";
        if (!selectedMode) {
            alert("Please select an option before running.");
            return;
        }

        let url = `/${selectedMode}`;

        let payload = {
            num_samples: 100 
        };

        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        })
        .then(response => {
            if (response.ok) {
                return response.blob();
            }
            throw new Error("Network response was not ok.");
        })
        .then(blob => {
            // Create an object URL for the image and show it
            let img = document.createElement('img');
            img.src = URL.createObjectURL(blob);
            document.getElementById("chart").innerHTML = "";  // Clear previous image
            document.getElementById("chart").appendChild(img);
            document.getElementById('run-info').textContent = "";

        })
        .catch(error => {
            console.error('There was a problem with the fetch operation:', error);
            document.getElementById('run-info').textContent = "";

        });
    });
});