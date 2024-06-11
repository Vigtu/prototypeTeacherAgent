import os
from crewai import Agent, Crew, Process, Task
from langchain_groq import ChatGroq 
import streamlit as st
import logging
from tasks import task_planner, task_searcher, task_reporter, task_integration
from backstory import backstory_planner, backstory_searcher, backstory_integration, backstory_reporter
from crewai_tools import BaseTool

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Set the environment
os.environ["OPENAI_API_BASE"] = 'https://api.groq.com/openai/v1'
os.environ["OPENAI_MODEL_NAME"] = 'llama3-70b-8192'
os.environ["OPENAI_API_KEY"] = '-----------------' 
os.environ["GROQ_API_KEY"] = '-----------------'

# Define the custom tool for reading text files
# O texto.txt foi criado apenas como placeholder, para testar se o código está funcionando
class TextFileReadTool(BaseTool):
    name: str = "TextFileReadTool"
    description: str = "Reads the content of a specified text file on 'texto.txt' and returns it."

    def _run(self, file_path: str = 'texto.txt') -> str:
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"

# Instantiate the text file read tool
text_file_read_tool = TextFileReadTool()

# Define the manager LLM
manager_llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

# Define the agents
planning_agent = Agent(
    role='Planner',
    goal='Streamline complex inquiries into organized, manageable components.',
    backstory=backstory_planner,
    verbose=True,
    cache=True,
    allow_delegation=True,
)

search_agent = Agent(
    role='Searcher',
    goal='Identify and retrieve essential data for sophisticated inquiries.',
    backstory=backstory_searcher,
    tools=[text_file_read_tool],
    verbose=True,
    cache=True,
    allow_delegation=True
)

integration_agent = Agent(
    role='Integration',
    goal="Organize and synthesize information from multiple sources",
    backstory=backstory_integration,
    verbose=True,
    cache=True,
    allow_delegation=True
)

reporting_agent = Agent(
    role='Reporter',
    goal="Communicate insights clearly, ensuring depth and accuracy for further exploration",
    backstory=backstory_reporter,
    verbose=True,
    cache=True,
    allow_delegation=True
)

def main(query):
    planning_task = Task(
        description=task_planner.format(query=query),
        expected_output='A detailed view of the sub-questions and their relationships to the main question and how to proceed with the investigation to answer the main question',
        agent=planning_agent,
    )

    search_task = Task(
        description=task_searcher.format(query=query),
        expected_output='Specific information and sources relevant to the sub-questions identified by the Planner Agent',
        agent=search_agent,
        tools=[text_file_read_tool],
        context=[planning_task]    
    )

    integration_task = Task(
        description=task_integration.format(query=query),
        expected_output='All the information gathered from the searcher agent organized and integrated with website links and references.',
        agent=integration_agent,
        context=[search_task, planning_task]
    )

    reporting_task = Task(
        description=task_reporter.format(query=query),
        expected_output='A clear, accurate, and concise response to the user, with references and website links to sources of information.',
        agent=reporting_agent,
        context=[integration_task, search_task, planning_task]
    )

    crew = Crew(
        agents=[planning_agent, search_agent, integration_agent, reporting_agent],
        process=Process.hierarchical,
        manager_llm=manager_llm,
        memory=True,
        tasks=[planning_task, search_task, integration_task, reporting_task]
    )

    output = crew.kickoff(inputs={'search_query': query, 'file_path': 'texto.txt'})
    return output

def test_text_file_read_tool_with_planner():
    query = input("Por favor, insira sua pergunta: ")
    logging.debug("Testing TextFileReadTool with Planner-defined keywords")
    result = main(query)
    logging.debug(f"Result: {result}")
    print(result)

def main_query():
    st.set_page_config(page_title="Crypto Teacher", page_icon=":book:")

    st.header("Crypto Teacher Response Generator")
    message = st.text_area("Ask a question about Crypto")

    if message:
        st.write("Generating the best response...")
        try:
            logging.debug("Calling main function with the query")
            result = main(message)
            logging.debug(f"Result: {result}")
            st.info(result)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Exception: {str(e)}")

if __name__ == '__main__':
    logging.debug("Starting app")
    test_text_file_read_tool_with_planner()
    # main_query()  # Descomente esta linha para usar a interface do Streamlit
