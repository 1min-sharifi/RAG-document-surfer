from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

LANGUAGE_MODEL = "gpt-4o"


def generate_answer(docs, question):
    system = """You are an assistant for question-answering tasks. Answer the question based on the documents passed to you. 
DO NOT use your information. Use three-to-five sentences maximum and keep the answer concise."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>"),
        ]
    )

    llm = ChatOpenAI(model=LANGUAGE_MODEL, temperature=0)
    info = "\n".join(f"<doc{i+1}>:\nContent:{doc.page_content}\n</doc{i+1}>\n" for i, doc in enumerate(docs))
    rag_chain = prompt | llm | StrOutputParser()

    answer = rag_chain.invoke({"documents":info, "question": question})
    return answer