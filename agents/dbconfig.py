from sqlalchemy import create_engine
from langchain.sql_database import SQLDatabase

def config_mysql_db(host, user, password, db_name):
    """Configure MySQL Database connection."""
    if not (host and user and password and db_name):
       return None
   
    db_engine = create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{db_name}")
    db = SQLDatabase(db_engine)
    return db