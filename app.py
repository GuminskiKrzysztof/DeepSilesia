import io
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
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

# Initialize the Flask application
app = Flask(__name__)

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

# Train Regressor
def train_regressor():
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

def train_classifier():

    iris_data = load_iris()
    features = iris_data.data
    labels = iris_data.target
    
    features = MinMaxScaler().fit_transform(features)
    
    num_features = features.shape[1]

    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
    feature_map.decompose().draw(output="mpl", style="clifford", fold=20)
    
    

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


# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for training classifier
@app.route('/train', methods=['POST'])
def train():
    img = train_classifier()
    return send_file(img, mimetype='image/png')

# Route for training regressor
@app.route('/regress', methods=['POST'])
def regress():
    img = train_regressor()
    return send_file(img, mimetype='image/png')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)