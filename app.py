import matplotlib.pyplot as plt
import numpy as np
import io
from flask import Flask, render_template, request, send_file
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_algorithms.optimizers import COBYLA

app = Flask(__name__)

algorithm_globals.random_seed = 42

# Funkcja do trenowania klasyfikatora kwantowego
def train_classifier():
    num_inputs = 2
    num_samples = 20
    X = 2 * algorithm_globals.random.random([num_samples, num_inputs]) - 1
    y01 = 1 * (np.sum(X, axis=1) >= 0)
    y = 2 * y01 - 1

    # Tworzenie kwantowej sieci neuronowej
    qc = QNNCircuit(num_qubits=2)
    estimator_qnn = EstimatorQNN(circuit=qc)
    
    # Tworzenie klasyfikatora
    estimator_classifier = NeuralNetworkClassifier(
        estimator_qnn, optimizer=COBYLA(maxiter=60)
    )
    
    # Trenowanie klasyfikatora
    estimator_classifier.fit(X, y)
    
    # Przewidywanie wyników
    y_predict = estimator_classifier.predict(X)
    
    # Rysowanie wyników
    plt.figure(figsize=(6, 4))
    for x, y_target, y_p in zip(X, y, y_predict):
        if y_target == 1:
            plt.plot(x[0], x[1], "bo")
        else:
            plt.plot(x[0], x[1], "go")
        if y_target != y_p:
            plt.scatter(x[0], x[1], s=200, facecolors="none", edgecolors="r", linewidths=2)
    plt.plot([-1, 1], [1, -1], "--", color="black")
    
    # Zapisanie wykresu
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return img

# Strona główna serwująca formularz HTML
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint do trenowania modelu
@app.route('/train', methods=['POST'])
def train():
    img = train_classifier()
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
