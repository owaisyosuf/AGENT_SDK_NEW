
from agents import Agent,Runner,AsyncOpenAI,set_tracing_disabled, set_default_openai_api,set_default_openai_client ,AgentOutputSchemaBase
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

class MovieReview(BaseModel):
    is_positive: bool
    genre: str
    summary: str

class OutputSchema(AgentOutputSchemaBase):
    def is_plain_text(self) -> bool:
        return False

    def name(delf) -> str:
        return "MovieReviews"
    
    def json_schema(self):
        review = {
            'is_positive': True, 
            'genre': 'sci-fi', 
            'summary': 'A thrilling and exciting sci-fi blockbuster with edge-of-the-seat action scenes and a captivating plot.'
        }
        return review
    
    def is_strict_json_schema(self) -> bool:
        return True

agent = Agent(
    name="MovieAnalyzer",
    instructions="Analyze the movie review and determine if it's positive, identify the genre, and provide a brief summary. Include any additional details like sentiment score or related genres if relevant.",
    model=global_model,
    output_type=OutputSchema
)

result = Runner.run_sync(
    starting_agent=agent,
    input="I loved the thrilling action scenes in this sci-fi blockbuster! The plot kept me on edge.",
)

print(result.final_output)