from langchain_community.utilities import  WikipediaAPIWrapper
from langchain_community.tools import   WikipediaQueryRun
from langchain.schema import Document

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

def wiki_search(state):
    """
    wiki search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---wikipedia---")
    print("---HELLO--")
    question = state["question"]
    print(question)

    # Wiki search
    docs = wiki_tool.invoke({"query": question})
    wiki_results = Document(page_content=docs)

    return {"documents": wiki_results, "question": question}