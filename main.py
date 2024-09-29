from flask import Flask, redirect, url_for, render_template, send_file, jsonify, request, send_from_directory
from models.linear_regression import LinearRegressionModel
from models.binary_classification import Binary_classification
from models.iris_classification import IrisQuantumClassifier
import os


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/manual")
def manual():
    return render_template("manual.html")

@app.route('/playground')
def index():
    return render_template('playground.html')


@app.route('/train', methods=['POST'])
def train():
    bc = Binary_classification()
    img = bc.train_classifier()
    return send_file(img, mimetype='image/png')

@app.route('/regress', methods=['POST'])
def regress():
    lr = LinearRegressionModel()
    img, execution_time = lr.execute_quantum_regression()
    return send_file(img, mimetype='image/png'), 200, {'Execution-Time': str(execution_time)}

@app.route('/iris', methods=['POST'])
def iris():
    ir = IrisQuantumClassifier()
    img, execution_time, train_score, test_score = ir.execute_classification()
    return send_file(img, mimetype='image/png'), 200, {'Execution-Time': str(execution_time)}

@app.route("/compare")
def compare():
    return render_template("compare.html")

@app.route('/train_quantum', methods=['POST'])
def train_q():
    data = request.get_json() 
    num_samples = int(data['num_samples'])  
    print(num_samples)
    bc = Binary_classification(num_samples)
    img, execution_time = bc.train_classifier()

    return send_file(img, mimetype='image/png'), 200, {'Execution-Time': str(execution_time)}

@app.route('/train_classical', methods=['POST'])
def train_classical():
    data = request.get_json()  
    num_samples = int(data['num_samples'])  
    
    bc = Binary_classification(num_samples)
    img, execution_time = bc.train_classical_classifier()

    return send_file(img, mimetype='image/png'), 200, {'Execution-Time': str(execution_time)}


if __name__ == "__main__":
    app.run(debug=True)
