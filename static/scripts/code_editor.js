const editor = CodeMirror.fromTextArea(document.getElementById("code"), {
	mode: { name: "python", version: 3, singleLineStringErrors: false },
	lineNumbers: true,
	indentUnit: 4,
	matchBrackets: true,
	theme: "dracula"
});

function loadFile(filePath) {
    fetch(filePath)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok ' + response.statusText);
            }
            return response.text();
        })
        .then(data => {
            // Ustaw zawartość pliku w edytorze
            editor.setValue(data);
        })
        .catch(error => {
            console.error('There was a problem with the fetch operation:', error);
        });
}

function saveToFile() {
	// Pobierz zawartość z edytora tekstowego (CodeMirror)
	const content = editor.getValue();

	// Utwórz obiekt Blob z zawartością edytora
	const blob = new Blob([content], { type: 'text/plain' });

	// Utwórz URL dla pliku Blob
	const fileUrl = URL.createObjectURL(blob);

	// Utwórz element <a> i ustaw go tak, by pobierał plik
	const a = document.createElement('a');
	a.href = fileUrl;
	a.download = 'code.py'; // nazwa pliku do pobrania
	a.click();

	// Zwolnij URL dla pliku Blob, aby oszczędzać pamięć
	URL.revokeObjectURL(fileUrl);
}

// const output = document.getElementById("output"); 

// const editor = CodeMirror.fromTextArea(document.getElementById("code"), { 
// 	mode: { 
// 		name: "python", 
// 		version: 3, 
// 		singleLineStringErrors: false
// 	}, 
// 	lineNumbers: true, 
// 	indentUnit: 4, 
// 	matchBrackets: true,
//     theme: "dracula"
// }); 

// // const filePath = '\\strings\\classification.txt';
// // const fs = require('fs');
// // // Odczytanie pliku asynchronicznie
// // fs.readFile(filePath, 'utf8', (err, data) => {
// //     if (err) {
// //         console.error('Error reading file:', err);
// //         return;
// //     }
// //     console.log('File content:', data); // Wyświetlenie zawartości pliku
// // });

// my_code = 
// `import datetime
// import io
// import numpy as np
// import matplotlib
// matplotlib.use('Agg') 
// import matplotlib.pyplot as plt
// from sklearn.model_selection import train_test_split
// from sklearn.linear_model import LogisticRegression, LinearRegression
// from sklearn.datasets import make_classification
// from qiskit import QuantumCircuit
// from qiskit.circuit import Parameter
// from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
// from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
// from qiskit_algorithms.utils import algorithm_globals

// from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
// from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
// from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
// from qiskit_machine_learning.circuit.library import QNNCircuit
// from sklearn.datasets import load_iris
// from sklearn.preprocessing import MinMaxScaler
// from flask import Flask, redirect, url_for, render_template, send_file, jsonify
// import base64
// import time
// from io import BytesIO 

// class Binary_classification:
//     def __init__(self, num_samples=100, random_seed=42, num_inputs=4, num_qubits=4):
//         self.random_seed = random_seed
//         self.num_samples = num_samples
//         self.num_inputs = num_inputs
//         self.num_qubits = num_qubits
//         print(num_samples)

//     def generate_data(self):
//         X = 2 * algorithm_globals.random.random((self.num_samples, self.num_inputs)) - 1
//         y = 2 * (np.sum(X, axis=1) >= 0).astype(int) - 1
//         return X, y

//     def train_class(self, classifier, X, y):
//         start_time = datetime.datetime.now()
//         classifier.fit(X, y)
//         end_time = datetime.datetime.now()
        
//         execution_time = (end_time - start_time).total_seconds()
//         y_predict = classifier.predict(X)
        
//         return execution_time, y_predict
    
//     def plot_results(self, X, y, y_predict, title):
//         for x, y_target, y_p in zip(X, y, y_predict):
//             color = "bo" if y_target == 1 else "go"
//             plt.plot(x[0], x[1], color)
//             if y_target != y_p:
//                 plt.scatter(x[0], x[1], s=200, facecolors="none", edgecolors="r", linewidths=2)

//         plt.plot([-1, 1], [1, -1], "--", color="black")
//         plt.title(title)

//         img = io.BytesIO()
//         plt.savefig(img, format='png')
//         img.seek(0)
//         plt.close()
//         return img

//     def train_classifier(self):
//         X, y = self.generate_data()
//         qc = QNNCircuit(num_qubits=self.num_qubits)
//         estimator_qnn = EstimatorQNN(circuit=qc)
//         classifier = NeuralNetworkClassifier(estimator_qnn, optimizer=COBYLA(maxiter=60))

//         execution_time, y_predict = self.train_class(classifier, X, y)
//         img = self.plot_results(X, y, y_predict, "Quantum Classification Result")

//         return img, execution_time`


// editor.setValue(my_code)
// // editor.setValue(`function findSequence(goal) {
// // function find(start, history) {
// // 	if (start == goal)
// // 	return history;
// // 	else if (start > goal)
// // 	return null;
// // 	else
// // 	return find(start + 5, "(" + history + " + 5)") ||
// // 			find(start * 3, "(" + history + " * 3)");
// // }
// // return find(1, "1");
// // }`); 
// output.value = "Initializing...\n"; 



// async function main() { 
// 	let pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.18.1/full/" }); 
// 	await pyodide.loadPackage(['numpy', 'matplotlib', 'qiskit']);
// 	// Pyodide ready 
// 	output.value += "Ready!\n"; 
// 	return pyodide; 
// }; 

// let pyodideReadyPromise = main(); 

// function addToOutput(s) { 
// 	output.value += ">>>" + s + "\n"; 
// } 

// async function evaluatePython() { 
// 	let pyodide = await pyodideReadyPromise; 
// 	try { 
// 		console.log(editor.getValue()) 
// 		let output = pyodide.runPython(editor.getValue()); 
// 		addToOutput(output); 
// 	} catch (err) { 
// 		addToOutput(err); 
// 	} 
// } 



// // const variableText = 'Nowa treść do zapisania';

// // fetch('/update-script', {
// //     method: 'POST',
// //     headers: {
// //         'Content-Type': 'application/json',
// //     },
// //     body: JSON.stringify({ content: variableText }),
// // })
// // .then(response => {
// //     if (response.ok) {
// //         console.log('Plik został zaktualizowany pomyślnie.');
// //     } else {
// //         console.error('Wystąpił błąd podczas aktualizacji pliku.');
// //     }
// // })
// // .catch(error => console.error('Błąd sieciowy:', error));
