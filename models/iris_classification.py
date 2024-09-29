import io
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import VQC
from sklearn.model_selection import train_test_split
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

start = time.time()
vqc.fit(train_features, train_labels)
elapsed = time.time() - start

train_score = vqc.score(train_features, train_labels)
test_score = vqc.score(test_features, test_labels)

print(train_score)
print(test_score)