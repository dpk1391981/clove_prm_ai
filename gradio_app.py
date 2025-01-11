import cassio
import gradio as gr
from langchain_groq import ChatGroq
from langchain.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Literal, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from agents.sql_agent import sql_agent
from agents.wikipedia import wiki_search
from agents.rag_agent import retrieve
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

# Environment Variables (can be loaded with a configuration file)
from dotenv import load_dotenv
import os

## Load environment variables
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID_MULTI_AGENT = os.getenv("ASTRA_DB_ID_MULTI_AGENT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASS = os.getenv("MYSQL_PASS")
MYSQL_DB = os.getenv("MYSQL_DB")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Initialize Cassandra/AstraDB
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID_MULTI_AGENT)

# Data Model
class RoueQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "wiki_search", "sql_agent"] = Field(
        ...,
        description="Choose to route it to Wikipedia, vectorstore, or a SQL agent.",
    )

class GraphState(TypedDict):
    question: str
    llm: Any
    dbconfig: dict
    generation: str
    callbacks: Any
    documents: List[str]
    pdf_documents: Any

# LLM Setup
api_key = OPENAI_API_KEY
model = 'gpt-3.5-turbo'
llm = ChatOpenAI(api_key=api_key, model=model, temperature=0)

# Gradio Interface Setup
def chat_with_ai(question, agents, api_key_type, query_limit, upload_files):
    try:
        pdf_documents = []
        if agents == 'RAG-PDFs' and upload_files:
            for uploaded_file in upload_files:
                tempdf = f"./tmp.pdf"
                with open(tempdf, "wb") as file:
                    file.write(uploaded_file.getvalue())
                    file_name = uploaded_file.name

                loader = PyPDFLoader(tempdf)
                docs = loader.load()
                pdf_documents.extend(docs)

        # Set model and query agent
        if api_key_type == "Groq API":
            api_key = GROQ_API_KEY
            model = 'llama-3.3-70b-versatile'
            llm = ChatGroq(groq_api_key=api_key, model=model, streaming=True)

        # Routing logic based on agent selection
        if agents == 'RAG-PDFs':
            return "Routing to vectorstore"
        elif agents == "Wikipedia":
            return "Routing to Wikipedia"
        else:
            return "Routing to SQL Agent"

    except Exception as e:
        return f"Error: {e}"

# Gradio interface layout
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## PRM AI Assistant")
        gr.Markdown("Type your question below and get answers from the relevant datasource!")

        # Select the LLM type
        api_key_type = gr.Radio(choices=["Open API", "Groq API"], label="Choose LLM Type")
        
        # Select the agent type
        agents = gr.Radio(choices=["SQL", "RAG-PDFs", "Wikipedia"], label="Select Agent")
        
        # File upload for PDFs (for RAG-PDFs)
        upload_files = gr.File(label="Upload PDFs", file_count="multiple", type="file")

        # Textbox for user input
        user_input = gr.Textbox(label="Ask your question")

        # Output response
        response_output = gr.Textbox(label="Response", interactive=False)

        # Process button to get the response
        def process_input(user_input, agents, api_key_type, query_limit, upload_files):
            response = chat_with_ai(user_input, agents, api_key_type, query_limit, upload_files)
            return response

        submit_btn = gr.Button("Submit")
        submit_btn.click(process_input, inputs=[user_input, agents, api_key_type, gr.Number(default=100), upload_files], outputs=response_output)

    return demo

# Launch the Gradio interface
demo = create_gradio_interface()
demo.launch()
