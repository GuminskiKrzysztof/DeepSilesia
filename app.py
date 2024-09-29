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
from sklearn.decomposition import PCA

# Initialize the Flask application
app = Flask(__name__)

# Set random seed for reproducibility
algorithm_globals.random_seed = 42

def train_classifier(num_qubits=16):
    start_time = datetime.datetime.now()  # Start timing
    num_inputs = 20
    num_samples = 20
    X = 2 * algorithm_globals.random.random([num_samples, num_inputs]) - 1

    # Dynamically set number of qubits if not provided
    if num_qubits is None:
        num_qubits = num_inputs

    # Optional: If more inputs than qubits, apply PCA to reduce dimensionality
    if num_inputs > num_qubits:
        pca = PCA(n_components=num_qubits)
        X = pca.fit_transform(X)
        print(f"Reduced input data from {num_inputs} to {num_qubits} dimensions using PCA.")

    # Binary classification labels
    y01 = 1 * (np.sum(X, axis=1) >= 0)  # in {0, 1}
    y = 2 * y01 - 1  # in {-1, +1}

    # Construct QNN with the QNNCircuit's default ZZFeatureMap and RealAmplitudes ansatz
    qc = QNNCircuit(num_qubits=num_qubits)
    estimator_qnn = EstimatorQNN(circuit=qc)

    # Create the classifier
    classifier = NeuralNetworkClassifier(estimator_qnn, optimizer=COBYLA(maxiter=60))

    # Train the classifier
    classifier.fit(X, y)
    
    # Measure execution time
    end_time = datetime.datetime.now()
    execution_time = (end_time - start_time).total_seconds()

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

    return img, execution_time

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


# Funkcja do trenowania klasyfikatora klasycznego
def train_classical_classifier():
    # Generowanie losowych danych do klasyfikacji
    num_samples = 100  # Liczba próbek
    num_features = 2   # Liczba cech
    num_classes = 2    # Liczba klas

    X, y = make_classification(n_samples=num_samples, n_features=num_features, n_classes=num_classes, n_informative=2, n_redundant=0, random_state=42)

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Tworzenie modelu regresji logistycznej
    model = LogisticRegression(max_iter=200)
    date3 = datetime.datetime.now()
    model.fit(X_train, y_train)
    date4 = datetime.datetime.now()
    diff2 = date4 - date3
    print(diff2)

    # Predykcja dla zbioru testowego
    y_predict = model.predict(X_test)

    # Przygotowanie wykresu dla wyników
    for i in range(len(X_test)):
        plt.scatter(X_test[i][0], X_test[i][1], c='blue' if y_predict[i] == y_test[i] else 'red')

    plt.title("Wynik klasyfikacji klasycznej (Regresja Logistyczna)")
    plt.xlabel("Cechy 1")
    plt.ylabel("Cechy 2")
    
    # Zapis wykresu do obrazu PNG
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return img

# Funkcja do trenowania regresora klasycznego
def train_classical_regressor():
    num_samples = 50
    X = np.linspace(-3, 3, num_samples).reshape(-1, 1)
    y = np.sin(X) + 0.1 * np.random.normal(size=X.shape)

    # Tworzenie modelu regresji liniowej
    model = LinearRegression()
    model.fit(X, y)

    # Predykcja za pomocą modelu
    X_pred = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_pred = model.predict(X_pred)

    # Przygotowanie wykresu z predykcjami
    plt.plot(X, y, "bo", label="Dane treningowe")  # Wykres punktów danych treningowych
    plt.plot(X_pred, y_pred, "g-", label="Predykcja modelu")  # Wykres predykcji modelu

    # Dodanie etykiet i legendy do wykresu
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Wynik regresji klasycznej (Regresja Liniowa)")
    plt.legend()

    # Zapis wykresu do obrazu w pamięci
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return img


# Route for the main page
@app.route('/')
def index():
    return render_template('index2.html')

# Route for training classifier
@app.route('/train_quantum', methods=['POST'])
def train():
    img, execution_time = train_classifier()
    return send_file(img, mimetype='image/png'), 200, {'Execution-Time': str(execution_time)}

# Route for training regressor
@app.route('/regress_quantum', methods=['POST'])
def regress():
    img = train_regressor()
    return send_file(img, mimetype='image/png')

# Route dla treningu klasyfikatora klasycznego
@app.route('/train_classical', methods=['POST'])
def train_classical():
    img = train_classical_classifier()
    return send_file(img, mimetype='image/png')

# Route dla treningu regresora klasycznego
@app.route('/regress_classical', methods=['POST'])
def regress_classical():
    img = train_classical_regressor()
    return send_file(img, mimetype='image/png')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5003)