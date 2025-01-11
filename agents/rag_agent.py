from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from langchain.schema import Document

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    
    
    print(state)
    if state["pdf_documents"]:
        docs = state["pdf_documents"]
    else:
        base_dir = os.getcwd()  # Replace with os.path.dirname(os.path.abspath(__file__)) for scripts
        tempdf = os.path.join(base_dir, "agents/pdfs/clove_dental.pdf")
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
    
    
    #history retriever prompt
    contextual_q_system_prompt=(
        "Given a chat history and the latest user question"
        "which might refrence context in the chat history"
        "formulate a standlone quesiton which can be understood"
        "without the chat history, Do not the answer of question"
        "just reformulate it if needed and otherwise return it as it"
    )
    
    contextual_q_prompt=ChatPromptTemplate.from_messages(
        [
            ("system", contextual_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    history_aware_retriver=create_history_aware_retriever(state["llm"], retriever,contextual_q_prompt)
    
    #Answer
    sytem_prompt=(
        "You are an assistent for question and answer the task"
        "use the following piece of retrieved context to answer"
        "the question , you don't now the answer, say that you"
        "don't know . Use three sentences maximum and keep the "
        "answer concise"
        "\n\n"
        "{context}"
    )
    
    qa_prompt=ChatPromptTemplate.from_messages(
        [
            ("system", sytem_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    
    question_answer_chain=create_stuff_documents_chain(state["llm"],qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriver,question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            state['get_session_history'],
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
            callbacks=[state["callbacks"]]
        )
    
    question = state["question"]
    
    documents = conversational_rag_chain.invoke(
                {"input": question},
                config={
                    "configurable": {"session_id": state["session_id"]}
                },
            )

    # Accessing components
    input_query = documents["input"]
    chat_history = documents["chat_history"]
    context = documents["context"]
    answer = documents["answer"]
    query_result = Document(page_content=answer)
    return {"documents": query_result, "question": question}