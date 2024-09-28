import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Konfiguracja ścieżki do katalogu z dokumentami tekstowymi
input_dir_path = 'my_texts'  # Zmiana ścieżki do katalogu
index_file = 'index.pkl'

# Krok 1: Ładowanie dokumentów tekstowych z katalogu
class TextLoader:
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def load(self):
        documents = []
        for file_name in os.listdir(self.dir_path):
            file_path = os.path.join(self.dir_path, file_name)
            if file_path.endswith(".txt"):  # Akceptowanie plików .txt
                with open(file_path, 'r', encoding='utf-8') as file:
                    documents.append(file.read())
        return documents

loader = TextLoader(input_dir_path)
docs = loader.load()

# Krok 2: Tworzenie lub ładowanie indeksu FAISS
def save_index(index, index_file):
    with open(index_file, 'wb') as f:
        pickle.dump(index, f)

def load_index(index_file):
    with open(index_file, 'rb') as f:
        return pickle.load(f)

# Używamy wielojęzycznego modelu do tworzenia embeddingów
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Generowanie embeddingów z dokumentów tekstowych
doc_embeddings = model.encode(docs)

# Jeśli nie ma indeksu, tworzony jest nowy
if not os.path.exists(index_file):
    # Tworzenie nowego indeksu FAISS
    doc_embeddings = np.array(doc_embeddings, dtype=np.float32)  # Konwertowanie na float32
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])  # L2 distance
    index.add(doc_embeddings)
    save_index(index, index_file)
else:
    # Ładowanie istniejącego indeksu
    index = load_index(index_file)

# Krok 3: Wyszukiwanie kontekstowe w dokumentach tekstowych na podstawie zapytania
def search_documents(query):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)  # Konwertowanie na float32
    D, I = index.search(query_embedding, k=3)  # Znajdź 3 najbliższe dokumenty
    return [docs[i] for i in I[0]] if I[0][0] != -1 else []  # Zwracamy pustą listę, jeśli nie ma wyników

# Pipeline do generowania odpowiedzi w języku polskim
qa_pipeline = pipeline("question-answering", model="allegro/herbert-base-cased", device=-1)  # Użyj CPU

# Generowanie odpowiedzi na podstawie zapytania i kontekstu
def generate_answer(query, context_str):
    if context_str:  # Upewnij się, że kontekst nie jest pusty
        result = qa_pipeline(question=query, context=context_str)
        return result['answer']
    return "Brak kontekstu do udzielenia odpowiedzi."

# Wywołanie odpowiedzi
while True:
    query = input("Podaj zapytanie (lub wpisz 'n', aby zakończyć): ")
    if query.lower() == "n":
        break

    # Szukanie kontekstu
    context = search_documents(query)
    context_str = "\n".join(context)

    # Generowanie odpowiedzi
    answer = generate_answer(query, context_str)
    print("Odpowiedź:", answer)
