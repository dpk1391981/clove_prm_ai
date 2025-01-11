import cassio
import streamlit as st
from langchain_groq import ChatGroq
from langchain.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Literal, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from agents.sql_agent import sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from agents.rag_agent import retrieve
from agents.wikipedia import wiki_search
from pprint import pprint

# from dotenv import load_dotenv
# import os

# ## Load environment variables
# load_dotenv()
# ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
# ASTRA_DB_ID_MULTI_AGENT = os.getenv("ASTRA_DB_ID_MULTI_AGENT")
# LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
# MYSQL_HOST = os.getenv("MYSQL_HOST")
# MYSQL_USER = os.getenv("MYSQL_USER")
# MYSQL_PASS = os.getenv("MYSQL_PASS")
# MYSQL_DB = os.getenv("MYSQL_DB")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")


ASTRA_DB_APPLICATION_TOKEN =   st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_ID_MULTI_AGENT =  st.secrets["ASTRA_DB_ID_MULTI_AGENT"]
LANGCHAIN_API_KEY =  st.secrets["LANGCHAIN_API_KEY"]
MYSQL_HOST =  st.secrets["MYSQL_HOST"]
MYSQL_USER =  st.secrets["MYSQL_USER"]
MYSQL_PASS =  st.secrets["MYSQL_PASS"]
MYSQL_DB =  st.secrets["MYSQL_DB"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


# Initialize Cassandra/AstraDB
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID_MULTI_AGENT)

# Data Model
class RoueQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "wiki_search", "sql_agent"] = Field(
        ...,
        description="Choose to route it to Wikipedia, vectorstore, or a SQL agent.",
    )
    
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

if 'store' not in st.session_state:
    st.session_state.store={}

class GraphState(TypedDict):
    question: str
    llm: Any
    dbconfig: dict
    generation: str
    callbacks: Any
    documents: List[str]
    pdf_documents: Any
    get_session_history: Any
    session_id: Any
    
# LLM Setup
api_key = OPENAI_API_KEY

model = 'gpt-3.5-turbo'
llm = ChatOpenAI(api_key=api_key, model=model, temperature=0)

# Input box for user question
api_key_type = st.sidebar.selectbox(
    "Choose LLM Type",
    ("Open API", "Groq API"),
)

query_limit = 0
# Input box for user question
agents = st.sidebar.selectbox(
    "Select Agent",
    ("SQL", "RAG-PDFs", "Wikipedia" ),
)
session_id=st.sidebar.text_input("Session ID", value="default_session")
if api_key_type == "Groq API":
    api_key = GROQ_API_KEY
    model = 'llama-3.3-70b-versatile'
    llm = ChatGroq(groq_api_key=api_key, model="llama-3.3-70b-versatile", streaming=True)
   
if agents == 'SQL':
    query_limit = st.sidebar.number_input(
        "Set maximum number of rows to fetch per query:",
        min_value=1,
        max_value=1000000,
        value=100,
        step=1,
    )
    
pdf_documents=[]
if agents == 'RAG-PDFs':
    upload_files=st.sidebar.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    for uploaded_file in upload_files:
        tempdf=f"./tmp.pdf"
        with open(tempdf, "wb") as file:
            file.write(uploaded_file.getvalue())  
            file_name=uploaded_file.name
            
        loader=PyPDFLoader(tempdf)
        docs=loader.load()
        pdf_documents.extend(docs)
    
# Display the selected option
st.markdown(f"Default LLM API : **{api_key_type}** and Default Model: **{model}**")

print(pdf_documents)
# Prompt Setup
system = """
You are an expert at routing a user question to a vectorstore or Wikipedia. 
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks. 
Use the vectorstore for questions on these topics. Otherwise, use Wikipedia.
"""
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])
question_routes = route_prompt | llm.with_structured_output(RoueQuery)

# Function to route the question
def route_question(state):
    print("---ROUTE QUESTION---")
    print(state)
    # question = state["question"]
    # source = question_routes.invoke({"question": question})
    if(agents == "RAG-PDFs"):
        return "vectorstore"
    elif(agents == "Wikipedia"):
       return "wiki_search"
   
    return "sql_agent"

# Workflow setup
workflow = StateGraph(GraphState)
routeNode = {
    "sql_agent": "sql_agent",
}

if agents == "RAG-PDFs":
    routeNode["vectorstore"] = "retrieve"  # Fixed typo 'retireve' to 'retrieve'
elif agents == "Wikipedia":
    routeNode["wiki_search"] = "wiki_search"

#Add Node
workflow.add_node("sql_agent", sql_agent)
if(agents == "RAG-PDFs"):
    workflow.add_node("retrieve", retrieve)
elif(agents == "Wikipedia"):
    workflow.add_node("wiki_search", wiki_search)
    
workflow.add_conditional_edges(
    START,
    route_question,
    routeNode,
)

workflow.add_edge("sql_agent", END)
if(agents == "RAG-PDFs"):
    workflow.add_edge("retrieve", END)
elif(agents == "Wikipedia"):
    workflow.add_edge("wiki_search", END)
    
app = workflow.compile()

# Streamlit Interface
def chat_with_ai(question):
    try:
        print(f"after chat: {pdf_documents}")
        inputs = {
            "question": question,
            "llm": llm,
            "dbconfig": {
                "host": MYSQL_HOST,
                "user": MYSQL_USER,
                "pass": MYSQL_PASS,
                "db_name": MYSQL_DB,
                "limit": query_limit,
            },
            "callbacks": StreamlitCallbackHandler(st.container()),
            "pdf_documents": pdf_documents,
            "get_session_history": get_session_history,
            "session_id": session_id
        }
        response = []
        
        for output in app.stream(inputs):
            for key, value in output.items():
                pprint(f"Node '{key}':")
                response.append(value)
            pprint("\n---\n")
        
        # Display the last result
        return value['documents'].page_content
       
    except ValueError as e:
        return f"Parsing Error: {e}"
    except Exception as e:
        return f"Error: {e}"

# Streamlit App Layout
title = "PRM AI Assistant"
if agents == "RAG-PDFs":
    title = "RAG Based AI Assistant"
elif agents == "Wikipedia":
    title = "Wikipedia Agent AI Tool"
    
st.title(title)
st.markdown("Type your question below and get answers from the relevant datasource!")
    
# Chat history initialization
if "messages" not in st.session_state or st.sidebar.button("Clear History"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
# Handle user input
user_input = st.chat_input("Ask from your local database!")
    
withType = st.chat_message("assistant")
if agents == "RAG-PDFs" or agents == "Wikipedia":
    withType = st.spinner("Wait we are generating response ...")
    
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    with withType:
        # streamlit_callback = StreamlitCallbackHandler(st.container())
        response = chat_with_ai(user_input)
        if response:
            st.success("Done!")
            st.markdown(f"**Response:** {response}")
        else:
            st.error("Error!")
            # st.markdown(f"**Response:** {response}")
