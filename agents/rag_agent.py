from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
import os

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tempdf = os.path.join(base_dir, "./pdfs/clove_dental.pdf")
    print(tempdf)
    loader = PyPDFLoader(tempdf)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)
    docs_split = text_splitter.split_documents(documents=docs)

    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    astra_vector_store = Cassandra(
        embedding=embeddings,
        table_name="multi_agents_tbl",
        session=None,
        keyspace=None
    )
    astra_vector_store.add_documents(documents=docs_split)

    # Create retriever
    retriever = astra_vector_store.as_retriever()
    question = state["question"]
    
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}