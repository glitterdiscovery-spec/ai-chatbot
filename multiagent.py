
from crewai import Agent, Task, Crew, LLM
import os



llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

joker_agent = Agent(
    role="spy",
    goal="create a funny joke",
    backstory="logical and real world ",
    llm=llm
)
joker_task = Task(
    description="tell a 5 line joke for kids that can make the child smile",
    expected_output="5 lines of joke",
    agent=joker_agent
)

joker_judge = Agent(
    role="Joke Judge",
    goal="Judge the joke",
    backstory="You are a joke judge that could rate the joke out of 10",
    llm=llm
)

joker_judge_task = Task(
    description="Rate the joke out of 10",
    expected_output="out of 10 if the joke is good give it a good remarks and if the joke is bad give it an average remarks",
    agent=joker_judge
)

crew = Crew(
    agents=[joker_agent, joker_judge],
    tasks=[joker_task, joker_judge_task],
    verbose=False
)

result = crew.kickoff()
print(result)