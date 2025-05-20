import re
import hashlib
from pathlib import Path
import json
from tqdm import tqdm 
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore

LANGUAGE_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
DATA_FOLDER = "./data"
HASH_FILE = "./faiss-index/dir-hash.json"
VECTORSTORE_FILE = "./faiss-index"


def get_retiever(path, k=2):
    return load_vectorstore(path).as_retriever(search_kwargs={"k": k})

def load_vectorstore(path):
    if folder_has_changed(path):
        print("Folder has changed, re-building vectorstore...")
        build_vectorstore(path)

    vectorstore = FAISS.load_local(VECTORSTORE_FILE, 
                                   embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL), 
                                   allow_dangerous_deserialization=True)
    return vectorstore


def build_vectorstore(path, chunk_size=1000, chunk_overlap=200):
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()

    
    documents = filter_docs(documents)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    vectorstore = prepare_vector_store(cleaned_texts)

    vectorstore.save_local(VECTORSTORE_FILE)


def prepare_vector_store(texts):
    vector_store = None  
    with ThreadPoolExecutor() as pool:  
        futures = [pool.submit(generate_hype, c) for c in texts]
        for f in tqdm(as_completed(futures), total=len(texts)):  
            text, vectors = f.result() 
            if vector_store == None:  
                vector_store = FAISS(
                    embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
                    index=faiss.IndexFlatL2(len(vectors[0])),
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )
            
            texts_with_embedding_vectors = [(text.page_content, vec) for vec in vectors]
            
            vector_store.add_embeddings(texts_with_embedding_vectors)  

    return vector_store


def generate_hype(text):
    llm = ChatOpenAI(temperature=0, model_name=LANGUAGE_MODEL)
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    question_gen_prompt = PromptTemplate.from_template(
        "Analyze the input text and generate essential questions that, when answered, \
        capture the main points of the text. Each question should be one line, \
        without numbering or prefixes.\n\n \
        Text:\n{chunk_text}\n\nQuestions:\n"
    )
    question_chain = question_gen_prompt | llm | StrOutputParser()

    questions = question_chain.invoke({"chunk_text": text}).replace("\n\n", "\n").split("\n")
    return text, embedding_model.embed_documents(questions)


def filter_docs(documents):
    filtered_docs = []

    prev_file = None
    for doc in documents:
        current_file = doc.metadata['source']
        if current_file != prev_file:
            found_intro = False
            prev_file = current_file
        
        if not found_intro:
            filtered_docs.append(doc)
        
        if finished_intro(doc):
            found_intro = True

    return filtered_docs


def finished_intro(doc)-> bool:
    return bool(re.search(r"\n2 ([^\d\n]+)\n", doc.page_content, re.UNICODE))


def replace_t_with_space(documents):
    for doc in documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
    return documents


def folder_has_changed(path):
    current_hash = get_folder_hash(path)
    previous_hash = load_previous_hash()

    if current_hash != previous_hash:
        save_current_hash(current_hash)
        return True  # Changes detected

    return False  # No changes


def get_folder_hash(folder):
    files = sorted(Path(folder).rglob("*"))  # Recursively include all files
    sha256_hash = hashlib.sha256()

    for file in files:
        if file.is_file():
            # Include filename relative to the data folder
            rel_path = file.relative_to(folder).as_posix()
            sha256_hash.update(rel_path.encode('utf-8'))

            # Read file content in binary mode
            with open(file, "rb") as f:
                while chunk := f.read(8192):  # Efficiently handle large files
                    sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def load_previous_hash():
    if Path(HASH_FILE).exists():
        with open(HASH_FILE, "r") as f:
            return json.load(f).get("hash")
    return ""


def save_current_hash(current_hash):
    with open(HASH_FILE, "w") as f:
        json.dump({"hash": current_hash}, f)
