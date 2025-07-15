
from agents import Agent,Runner,AsyncOpenAI,set_tracing_disabled, set_default_openai_api,set_default_openai_client ,AgentOutputSchema
from pydantic import BaseModel
from dotenv import load_dotenv 
import os
import asyncio
load_dotenv()
set_tracing_disabled(True)
set_default_openai_api("chat_completions")


api_key=os.getenv("GEMINI_API_KEY")

if not api_key:
  raise ValueError("api key is not found")

external_client=AsyncOpenAI(
  api_key=api_key,
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

set_default_openai_client(external_client)
global_model="gemini-2.0-flash"

class math_type(BaseModel):
    is_math: bool
    expresstion: str
    answer: float
    steps:str

agent = Agent(
    name="MovieAnalyzer",
    instructions="Solve the math expression. Provide the final answer and explain the steps clearly.",
    output_type=AgentOutputSchema(math_type, strict_json_schema=False),
    model=global_model,
)

result = Runner.run_sync(
    starting_agent=agent,
    input="who is the founder of pakistan  ",
)

print(result.final_output)