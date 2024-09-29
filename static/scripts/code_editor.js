const output = document.getElementById("output"); 

const editor = CodeMirror.fromTextArea(document.getElementById("code"), { 
	mode: { 
		name: "python", 
		version: 3, 
		singleLineStringErrors: false
	}, 
	lineNumbers: true, 
	indentUnit: 4, 
	matchBrackets: true,
    theme: "dracula"
}); 

// const filePath = '\\strings\\classification.txt';
// const fs = require('fs');
// // Odczytanie pliku asynchronicznie
// fs.readFile(filePath, 'utf8', (err, data) => {
//     if (err) {
//         console.error('Error reading file:', err);
//         return;
//     }
//     console.log('File content:', data); // Wyświetlenie zawartości pliku
// });

my_code = 
`import io
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

# Set random seed for reproducibility
algorithm_globals.random_seed = 42

# Train Classifier
def train_classifier():
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

# Run the Flask app
if __name__ == '__main__':
    train_classifier()`


editor.setValue(my_code)
// editor.setValue(`function findSequence(goal) {
// function find(start, history) {
// 	if (start == goal)
// 	return history;
// 	else if (start > goal)
// 	return null;
// 	else
// 	return find(start + 5, "(" + history + " + 5)") ||
// 			find(start * 3, "(" + history + " * 3)");
// }
// return find(1, "1");
// }`); 
output.value = "Initializing...\n"; 

async function main() { 
	let pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.18.1/full/" }); 
	await pyodide.loadPackage(['numpy', 'matplotlib', 'IPython']);
	// Pyodide ready 
	output.value += "Ready!\n"; 
	return pyodide; 
}; 

let pyodideReadyPromise = main(); 

function addToOutput(s) { 
	output.value += ">>>" + s + "\n"; 
} 

async function evaluatePython() { 
	let pyodide = await pyodideReadyPromise; 
	try { 
		console.log(editor.getValue()) 
		let output = pyodide.runPython(editor.getValue()); 
		addToOutput(output); 
	} catch (err) { 
		addToOutput(err); 
	} 
} 



