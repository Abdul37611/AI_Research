import os
from crewai import Agent, Task, Process, Crew
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

researcher = Agent(
    role="Researcher",
    goal="Research new AI insights",
    backstory="You are an AI research assistant",
    verbose=True, 
    allow_delegation=True,
    )

general = Agent(
    role="General Assistant",
    goal="Provide assistance in general tasks",
    backstory="You are a general assistant, capable of handling a variety of tasks",
    verbose=True, 
    allow_delegation=False,
    )

class TextData(BaseModel):
  task: str
  model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "task": "What is the difference between a convolutional neural network and a recurrent neural network?",
                }
            ]
        }
    }

@app.post("/agents")
async def process_text(request: TextData):
  """
  An AI researcher agent that will assist you in your research.
  """
  task = Task(description = request.task, agent = researcher)

  crew = Crew(
    agents=[researcher,general],
    tasks=[task],
    verbose=2,
    process=Process.sequential
  )

  result = crew.kickoff()

  return {"data": result}
