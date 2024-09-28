import os
import pickle
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Konfiguracja ścieżki do katalogu z dokumentami PDF
input_dir_path = 'my_pdf'
index_file = 'index.pkl'

# Krok 1: Ładowanie dokumentów PDF z katalogu za pomocą pdfplumber
class PDFPlumberLoader:
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def load(self):
        documents = []
        for file_name in os.listdir(self.dir_path):
            file_path = os.path.join(self.dir_path, file_name)
            if file_path.endswith(".pdf"):
                with pdfplumber.open(file_path) as pdf:
                    texts = []
                    for page in pdf.pages:
                        texts.append(page.extract_text())
                    documents.append("\n".join(texts))
        return documents

loader = PDFPlumberLoader(input_dir_path)
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

# Generowanie embeddingów z dokumentów PDF
doc_embeddings = model.encode(docs)

# Jeśli nie ma indeksu, tworzony jest nowy
if not os.path.exists(index_file):
    # Tworzenie nowego indeksu FAISS
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])  # L2 distance
    index.add(doc_embeddings)
    save_index(index, index_file)
else:
    # Ładowanie istniejącego indeksu
    index = load_index(index_file)

# Krok 3: Wyszukiwanie kontekstowe w PDF na podstawie zapytania
def search_documents(query):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=3)  # Znajdź 3 najbliższe dokumenty
    return [docs[i] for i in I[0]]

# Pipeline do generowania odpowiedzi w języku polskim
qa_pipeline = pipeline("question-answering", model="allegro/herbert-base-cased", device = 0)

# Generowanie odpowiedzi na podstawie zapytania i kontekstu
def generate_answer(query, context_str):
    result = qa_pipeline(question=query, context=context_str)
    return result['answer']

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
