from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv 
import os

load_dotenv()

topic = input("Enter topic: ")

llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY")
)

search_tool = SerperDevTool()

researcher = Agent(
    role="Research Assistant",
    goal=f"Find 3 interesting facts about {topic}",
    backstory="You are a professional research assistant.",
    llm=llm,
    tools=[search_tool],
    verbose=True
)

research_task = Task(
    description=f'''Search the web and find 3 interesting facts about {topic}.''',
    expected_output="A bulleted list containing 3 interesting facts.",
    agent=researcher,
)

crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True
)

result = crew.kickoff(inputs={"topic": topic})
print(result)
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True,
    memory=True,
    output_log_file="crew_logs.txt",
    embedder={
        "provider": "sentence-transformer",
        "config": {
            "model": "all-MiniLM-L6-v2"
        }
    }
)

result = crew.kickoff()
print(result)