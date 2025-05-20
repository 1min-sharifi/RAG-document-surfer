from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

LANGUAGE_MODEL = "gpt-4o"

class GradeDocuments(BaseModel):

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def get_relevant_docs(docs, question):
    llm = ChatOpenAI(model=LANGUAGE_MODEL, temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    relevant_docs = []
    for doc in docs:
        res = retrieval_grader.invoke({"document": doc, "question": question})
        if res.binary_score == 'yes':
            relevant_docs.append(doc)
    return relevant_docs
