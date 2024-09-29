import numpy as np
import json

# Generate a dummy initial point with random weights, for example
# Adjust the size according to the expected number of weights in your model
dummy_initial_weights = np.random.rand(10).tolist()  # Replace 10 with the required size

# Save it to a JSON file
with open("11_qcnn_initial_point.json", "w") as f:
    json.dump(dummy_initial_weights, f)

print("Dummy initial point file created as '11_qcnn_initial_point.json'")