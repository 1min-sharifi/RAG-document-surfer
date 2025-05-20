from typing import List
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

LANGUAGE_MODEL = "gpt-4o"


class HighlightDocuments(BaseModel):
    """Return the specific part of a document used for answering the question."""

    id: List[str] = Field(
        ...,
        description="List of id of docs used to answers the question"
    )

    highlight: List[str] = Field(
        ...,
        description="List of direct segements from used documents that answers the question"
    )


def get_highlights(docs, question, answer):
    llm = ChatOpenAI(model=LANGUAGE_MODEL, temperature=0)

    parser = PydanticOutputParser(pydantic_object=HighlightDocuments)

    system = """You are an advanced assistant for document search and retrieval. You are provided with the following:
1. A question.
2. A generated answer based on the question.
3. A set of documents that were referenced in generating the answer.

Your task is to identify and extract the exact inline segments from the provided documents that directly correspond to the content used to 
generate the given answer. The extracted segments must be verbatim snippets from the documents, ensuring a word-for-word match with the text 
in the provided documents.

Ensure that:
- (Important) Each segment is an exact match to a part of the document and is fully contained within the document text.
- The relevance of each segment to the generated answer is clear and directly supports the answer provided.
- (Important) If you didn't used the specific document don't mention it.

Used documents: <docs>{documents}</docs> \n\n User question: <question>{question}</question> \n\n Generated answer: <answer>{generation}</answer>

<format_instruction>
{format_instructions}
</format_instruction>
"""

    prompt = PromptTemplate(
        template= system,
        input_variables=["documents", "question", "generation"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    doc_lookup = prompt | llm | parser
    info  = "\n".join(f"<doc{i+1}>:\nContent:{doc.page_content}\n</doc{i+1}>\n" for i, doc in enumerate(docs))
    lookup_response = doc_lookup.invoke({"documents":info, "question": question, "generation": answer})
    return lookup_response
