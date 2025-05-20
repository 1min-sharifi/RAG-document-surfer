# RAG Document Surfer

This repository implements a reliable Retrieval-Augmented Generation (RAG) system for answering user queries based on a set of academic publications. To improve efficiency and response time, only the **abstract** and **introduction** of each document are used.

## 🔍 System Overview

The system follows this pipeline:

1. **Document Retrieval**  
   A vectorstore is built using the **HyPE** method to retrieve documents relevant to the query.

2. **Relevance Filtering**  
   An LLM filters out irrelevant documents to refine the context.

3. **Answer Generation**  
   The filtered documents and the query are passed to another LLM to generate an answer.

4. **Factuality Validation**  
   A separate LLM checks whether the generated answer is factually supported by the provided documents.

5. **Evidence Highlighting**  
   A final LLM identifies small evidence snippets linking the query and answer for transparency.

## 🛠 Requirements

This project primarily uses **LangChain** and **Pydantic AI**. To install all dependencies, run:

```bash
pip install -r requirements.txt
```
## 🚀 Running the Code

After installing the dependencies, run:

```bash
python main.py
```
The script will prompt you to enter your **OpenAI API key**. It will then build the vectorstore (if it doesn't already exist) and launch an interactive interface for querying the system.
