from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def build_rag_chain(retriever):
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    prompt = ChatPromptTemplate.from_template(
        "Use the context below to answer the question:\n\n{context}\n\nQuestion: {input}"
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain