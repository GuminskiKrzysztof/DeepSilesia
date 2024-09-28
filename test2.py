import requests

def query_ollama(model, prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()

# Example usage
model = "llama3:8b-instruct-q4_1"
prompt = "Explain quantum machine learning"
response = query_ollama(model, prompt)
print(response['response'])