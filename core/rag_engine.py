import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.vector_store import get_retriever
# from langchain_chroma import Chroma 
import chromadb

def get_llm():
    return ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite", google_api_key = os.getenv("GEMINI_API_KEY"))

# def format_docs(docs):
#     return "\n\n".join([doc.page_content for doc in docs])

# vector_store = build_vector_store(transcript)
def build_rag_chain(question: str, vector_store: chromadb) -> str:

    retrieved_docs = get_retriever(vector_store=vector_store, query=question)

    context = "\n".join([doc["document"] for doc in retrieved_docs]) if retrieved_docs else ""
    
    if not context:
        print("we found no relevant context for the given query")
    
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(

        [(
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ]
    )

    #full LCEL Rag pipeline 
    
    rag_chain = prompt | llm | StrOutputParser()
    
    return rag_chain.invoke({"context": context, "question": question})



# def ask_question(rag_chain, question:str) -> str:
#     print(f"Question : {question}")
#     answer = rag_chain.invoke({"question": question})
#     print(f"answer :{answer}")
#     return answer


