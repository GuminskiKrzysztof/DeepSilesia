import io
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
import pandas as pd
import seaborn as sns
import time
import base64

np.random.seed(123)

iris_data = load_iris()
features = iris_data.data
labels = iris_data.target

df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df["class"] = pd.Series(iris_data.target)

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, train_size=0.8, random_state=123
)

num_features = features.shape[1]
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
ansatz = RealAmplitudes(num_qubits=num_features, reps=3)

optimizer = COBYLA(maxiter=100)

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train_vqc', methods=['POST'])
def train_vqc():
    start = time.time()
    vqc.fit(train_features, train_labels)
    elapsed = time.time() - start

    train_score = vqc.score(train_features, train_labels)
    test_score = vqc.score(test_features, test_labels)

    plt.figure()
    plt.bar(['Training Score', 'Test Score'], [train_score, test_score], color=['blue', 'orange'])
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Model VQC Scores')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    
    result = {
        "training_time": round(elapsed, 2),
        "train_score": f"{train_score:.2f}",
        "test_score": f"{test_score:.2f}",
        "image": img_base64  
    }
    return jsonify(result)

@app.route('/train_classic', methods=['POST'])
def train_classic():
    classifier = LogisticRegression(max_iter=200)
    start = time.time()
    classifier.fit(train_features, train_labels)
    elapsed = time.time() - start

    train_score = classifier.score(train_features, train_labels)
    test_score = classifier.score(test_features, test_labels)

    plt.figure()
    plt.bar(['Training Score', 'Test Score'], [train_score, test_score], color=['green', 'red'])
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Classic Model Scores')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    result = {
        "training_time": round(elapsed, 2),
        "train_score": f"{train_score:.2f}",
        "test_score": f"{test_score:.2f}",
        "image": img_base64  
    }
    return jsonify(result)

@app.route('/plot')
def plot():
    plt.figure()
    sns.pairplot(df, hue='class', palette='tab10')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return jsonify({"image": img_base64})

if __name__ == '__main__':
    app.run(debug=True, port=5005)
