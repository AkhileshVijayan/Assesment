# from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
# from llama_index.core.selectors import LLMSingleSelector
# from llama_index.core.tools import QueryEngineTool
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from pathlib import Path
import os
import re
from typing import Tuple
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# llm = Ollama(model="llama3.1:8b",request_timeout=3000.0)
llm = OpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
def cleaned_tool_name(name):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)


def create_doc_tools(
        document_fp: str,
        doc_name: str,
        verbose: bool = True,
) -> Tuple[QueryEngineTool, QueryEngineTool]:
    documents = SimpleDirectoryReader(input_files=[document_fp]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    # Settings.llm = Ollama(model="llama3.1:8b", request_timeout=120.0)
    # Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text:latest")
    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)

    # summary index
    summary_index = SummaryIndex(nodes)
    # vector store index
    vector_index = VectorStoreIndex(nodes)

    # summary query engine
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )

    # vector query engine
    vector_query_engine = vector_index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        name=cleaned_tool_name(f"{doc_name}_summary_query_engine_tool"),
        query_engine=summary_query_engine,
        description=(
            f"Useful for summarization questions related to the {doc_name}."
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        name=cleaned_tool_name(f"{doc_name}_vector_query_engine_tool"),
        query_engine=vector_query_engine,
        description=(
            f"Useful for retrieving specific context from the the {doc_name}."
        ),
    )

    return vector_tool, summary_tool


def retrive_qa_documents():
    directory = "cubet_pdfs"
    papers = [f"./{directory}/{filename}" for filename in os.listdir(directory) if filename.endswith('.pdf')]
    paper_to_tools_dict = {}
    for paper in papers:
        print(f"Creating {paper} tool")
        path = Path(paper)
        vector_tool, summary_tool = create_doc_tools(doc_name=path.stem, document_fp=path)
        paper_to_tools_dict[path.stem] = [vector_tool, summary_tool]
    initial_tools = [t for paper in papers for t in paper_to_tools_dict[Path(paper).stem]]
    print(str(initial_tools))
    agent_worker = FunctionCallingAgentWorker.from_tools(
        initial_tools,
        llm=llm,
        verbose=True
    )
    return agent_worker

agent_worker  = retrive_qa_documents()
agent = AgentRunner(agent_worker)



def q_a_chatbot():
    print("Welcome to the document Q&A Chatbot!")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        try:

            # Sending user's question to the agent
            response = agent.query(user_input)
            # Output the response from the agent
            print("Chatbot:", str(response))
        except Exception as e:
            print("Chatbot: Sorry, I couldn't process your request.")
            print(f"Error: {e}")


if __name__ == "__main__":
    q_a_chatbot()