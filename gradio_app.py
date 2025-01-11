import gradio as gr
# from langchain.callbacks import GradioCallbackHandler
from langchain_groq import ChatGroq
import cassio
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from agents.rag_agent import retrieve
from agents.sql_agent import sql_agent
from agents.wikipedia import wiki_search
from dotenv import load_dotenv
import os
from pprint import pprint

# Load environment variables
load_dotenv()
ASTRA_DB_APPLICATION_TOKEN =  os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID_MULTI_AGENT = os.getenv("ASTRA_DB_ID_MULTI_AGENT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASS = os.getenv("MYSQL_PASS")
MYSQL_DB = os.getenv("MYSQL_DB")

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
    llm: ChatGroq
    dbconfig: dict
    generation: str
    documents: List[str]
    
    
# LLM Setup
api_key = "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"
llm = ChatGroq(groq_api_key=api_key, model="llama3-8b-8192", streaming=True)
    
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
    return "sql_agent"
    # if source.datasource == "wiki_search":
    #     print("---ROUTE QUESTION TO Wiki SEARCH---")
    #     return "wiki_search"
    # elif source.datasource == "vectorstore":
    #     print("---ROUTE QUESTION TO RAG---")
    #     return "vectorstore"
    # elif source.datasource == "sql_agent":
    #     return "sql_agent"



# Workflow setup
workflow = StateGraph(GraphState)
workflow.add_node("sql_agent", sql_agent)
# workflow.add_node("retrieve", retrieve)
# workflow.add_node("wiki_search", wiki_search)

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "sql_agent": "sql_agent",
        # "vectorstore": "retrieve",
        # "wiki_search": "wiki_search",
    },
)
workflow.add_edge("sql_agent", END)
# workflow.add_edge("retrieve", END)
# workflow.add_edge("wiki_search", END)
app = workflow.compile()

# Gradio Interface
def chat_with_ai(question):
    try:
        inputs = {"question": question, "llm": llm, "dbconfig": {
            "host": MYSQL_HOST,
            "user": MYSQL_USER,
            "pass": MYSQL_PASS,
            "db_name": MYSQL_DB,
            "limit": 100
        }}
        response = []
        
        for output in app.stream(inputs):
            for key, value in output.items():
                response.append(value)
        
        # Display the last result
        if response:
            documents = response[-1]
            return documents["documents"].page_content
        else:
            return "No response received from the workflow."
    except ValueError as e:
        return f"Parsing Error: {e}"
    except Exception as e:
        return f"Error: {e}"

# Gradio UI
with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.red, secondary_hue=gr.themes.colors.pink)) as demo:
    gr.Markdown("## PRM AI Assistant")
    with gr.Row():
        input_box = gr.Textbox(label="Ask your question", placeholder="Type your question here...")
        output_box = gr.Textbox(label="Response")
    
    submit_button = gr.Button("Submit")
    submit_button.click(chat_with_ai, inputs=[input_box], outputs=[output_box])

# Launch the Gradio App
demo.launch()
