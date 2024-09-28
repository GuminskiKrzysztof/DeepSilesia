from flask import Flask, redirect, url_for, render_template, send_file
from models.linear_regression import Linear_regression
from models.binary_classifier import Binary_classifier

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
    bc = Binary_classifier()
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
    bc = Binary_classifier()
    img = bc.train_classifier()
    return send_file(img, mimetype='image/png')


@app.route('/train_classical', methods=['POST'])
def train_classical():
    bc = Binary_classifier()
    img = bc.train_classical_classifier()
    return send_file(img, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)