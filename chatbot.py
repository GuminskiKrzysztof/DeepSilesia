from transformers import pipeline

# Pipeline do generowania odpowiedzi
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Zapytanie do modelu z kontekstem
def generate_answer(query, context_str):
    result = qa_pipeline(question=query, context=context_str)
    return result['answer']

# Wywołanie odpowiedzi
while True:
    query = input("Podaj zapytanie (lub wpisz 'n' aby zakończyć): ")
    if query.lower() == "n":
        break

    # Szukanie kontekstu
    context = search_documents(query)
    context_str = "\n".join(context)

    # Generowanie odpowiedzi
    answer = generate_answer(query, context_str)
    print("Odpowiedź:", answer)
