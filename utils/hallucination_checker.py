from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

LANGUAGE_MODEL = "gpt-4o"


class GradeHallucinations(BaseModel):
    binary_score: str = Field(
        ...,
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


def hallucination_checker(docs, answer):
    llm = ChatOpenAI(model=LANGUAGE_MODEL, temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n <facts>{documents}</facts> \n\n LLM generation: <generation>{generation}</generation>"),
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm_grader
    info = "\n".join(f"<doc{i+1}>:\nContent:{doc.page_content}\n</doc{i+1}>\n" for i, doc in enumerate(docs))
    response = hallucination_grader.invoke({"documents": info, "generation": answer})
    return response.binary_score
