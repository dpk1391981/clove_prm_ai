from langchain.prompts import SemanticSimilarityExampleSelector,PromptTemplate,FewShotPromptTemplate
from few_shots import few_shots
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from agents.dbconfig import config_mysql_db
from langchain.agents.agent_types import AgentType
from langchain.schema import Document

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples=few_shots,
    embeddings=embeddings,
    vectorstore_cls=FAISS,
    k=2
)

mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".
    
    Use the following format:
    
    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    
    If the question is about **clinics**, return **ALL** these tables:
    - "facility"
    - "users"
    
    ### Category Mappings:
    1. **Category: receipts**
    - Table: receipt
  
    No pre-amble.
    """
    
example_prompt = PromptTemplate(
    input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
    template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
)

few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
    )

def sql_agent(state):
    print("---sql agent---")
    question = state["question"]
    print(question)
    print(state)
    
    db = config_mysql_db(state["dbconfig"]["host"], state["dbconfig"]["user"], state["dbconfig"]["pass"], state["dbconfig"]["db_name"])
    toolkit = SQLDatabaseToolkit(llm=state["llm"], db=db)
    
    limited_query = f"{question} LIMIT {state['dbconfig']['limit']}"
    
    # sql agent
    agent = initialize_agent(
        tools=toolkit.get_tools(),
        llm=state["llm"],
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,  # Enable parsing error handling
        prompt=few_shot_prompt,  # Use the FewShotPromptTemplate
    )
    
    docs = agent.run(limited_query, callbacks=[state["callbacks"]])
    #print(docs["summary"])
    query_result = docs
    query_result = Document(page_content=query_result)

    return {"documents": query_result, "question": question}