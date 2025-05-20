import os
from utils.retriever import get_retiever
from utils.relevancy_checker import get_relevant_docs
from utils.answer_generator import generate_answer
from utils.hallucination_checker import hallucination_checker
from utils.highlighter import get_highlights

DATA_FOLDER = "./data"
HASH_FILE = "./faiss_index/dir-hash.json"
VECTORSTORE_FILE = "./faiss_index/faiss_index"


def main():
    api_key = input("Enter your OpenAI API key:")
    os.environ["OPENAI_API_KEY"] = api_key

    retriever = get_retiever(DATA_FOLDER, k=5)
    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        docs = retriever.invoke(query)
        docs = list({doc.page_content: doc for doc in docs}.values())
        docs = get_relevant_docs(docs, query)
        answer = generate_answer(docs, query)
        hallucination = hallucination_checker(docs, answer)
        highlights = get_highlights(docs, query, answer)

        print("question:", query)
        print("Relevant documents:")
        for doc in docs:
            print(doc.page_content, '\n', '-'*50)
        print("LLM Answer:", answer)
        print("Factuality check:", hallucination)
        for id, highlight in zip(highlights.id, highlights.highlight):
            print(f"ID: {id}\nText Segment: {highlight}\n")


if __name__ == "__main__":
    main()
