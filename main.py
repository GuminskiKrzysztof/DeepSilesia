from flask import Flask, redirect, url_for, render_template, send_file, jsonify, request
from models.linear_regression import Linear_regression
from models.binary_classification import Binary_classification
import os


app = Flask(__name__)



@app.route("/")
def home():
    return render_template("index.html")

    

@app.route("/manual")
def manual():
    return render_template("manual.html")

# Route for the main page
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
    lr = Linear_regression()
    img = lr.train_regression()
    return send_file(img, mimetype='image/png')

@app.route("/compare")
def compare():
    return render_template("compare.html")


@app.route('/train_quantum', methods=['POST'])
def train_q():
    data = request.get_json()  # Pobranie danych JSON
    num_samples = int(data['num_samples'])  # Odczytanie wartości num_samples
    print(num_samples)
    # Inicjalizacja klasyfikatora z odpowiednią liczbą próbek
    bc = Binary_classification(num_samples)
    img, execution_time = bc.train_classifier()

    # Zwróć obraz i czas wykonania
    return send_file(img, mimetype='image/png'), 200, {'Execution-Time': str(execution_time)}



@app.route('/train_classical', methods=['POST'])
def train_classical():
    data = request.get_json()  # Pobranie danych JSON
    num_samples = int(data['num_samples'])  # Odczytanie wartości num_samples
    
    # Inicjalizacja klasyfikatora z odpowiednią liczbą próbek
    bc = Binary_classification(num_samples)
    img, execution_time = bc.train_classical_classifier()

    # Zwróć obraz i czas wykonania
    return send_file(img, mimetype='image/png'), 200, {'Execution-Time': str(execution_time)}


if __name__ == "__main__":
    app.run(debug=True)