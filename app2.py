from flask import Flask, render_template, jsonify, url_for
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit_algorithms.utils import algorithm_globals

from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit

import os

app = Flask(__name__)

# Folder do przechowywania wygenerowanych obrazów
IMG_FOLDER = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

# Funkcja generująca obraz obwodu kwantowego
@app.route('/generate_circuit')
def generate_circuit():
    # Tworzymy obwód kwantowy
    qc = QuantumCircuit(2)
    qc.h(0)  # Bramki Hadamarda
    qc.cx(0, 1)  # Bramki CNOT
    
    # Zapisz obwód do pliku jako obraz
    fig = qc.draw(output='mpl')
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'circuit.png')
    fig.savefig(image_path)
    
    return jsonify({'image_url': url_for('static', filename='images/circuit.png')})


# Strona główna
@app.route('/')
def home():
    return render_template('index3.html')

if __name__ == '__main__':
    # Utwórz folder na obrazy, jeśli nie istnieje
    if not os.path.exists(IMG_FOLDER):
        os.makedirs(IMG_FOLDER)
    app.run(debug=True, port=5004)
