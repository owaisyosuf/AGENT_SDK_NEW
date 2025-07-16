
from agents import Agent,Runner,AsyncOpenAI,set_tracing_disabled, set_default_openai_api,set_default_openai_client ,AgentOutputSchemaBase,enable_verbose_stdout_logging
from pydantic import BaseModel
from dotenv import load_dotenv 
import os
import asyncio
load_dotenv()
set_tracing_disabled(True)
set_default_openai_api("chat_completions")
# enable_verbose_stdout_logging()



api_key=os.getenv("GEMINI_API_KEY")

if not api_key:
  raise ValueError("api key is not found")

external_client=AsyncOpenAI(
  api_key=api_key,
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

set_default_openai_client(external_client)
global_model="gemini-2.0-flash"

# class MovieReview(BaseModel):
#     is_positive: bool
#     genre: str
#     summary: str

# class OutputSchema(AgentOutputSchemaBase):
#     def is_plain_text(self) -> bool:
#         return False

#     def name(delf) -> str:
#         return "Math review"
    
#     def json_schema(self):
#         review = {
#             'is_math': True, 
#             # 'expression': str, 
#             # 'summary': str,
#             # "result": float
#         }
#         return review
    
#     def is_strict_json_schema(self) -> bool:
#         return True

# agent = Agent(
#     name="math checker",
#     instructions="Solve the math expression. Provide the final answer and explain the steps clearly.",
#     model=global_model,
#     output_type=OutputSchema
# )

# result = Runner.run_sync(
#     starting_agent=agent,
#     input="what is 2+2",
# )

# print(result.final_output)

class MathOutput(AgentOutputSchemaBase , BaseModel):
    result: int
    is_even: bool
    operator_count: int

    def is_plain_text(self) -> bool:
        return True

    def name(self) -> str:
        return "MathEvaluator"

    def json_schema(self):
        return {
            "result": 20,
            "is_even": True,
            "operator_count": 2
        }

    def is_strict_json_schema(self) -> bool:
        return False
   
    @classmethod
    def validate_json(cls, json_data: str) -> "MathOutput":
        from json import loads
        data = loads(json_data)
        return cls(**data)

    # 💡 Required for Pydantic compatibility
    model_config = {
        "arbitrary_types_allowed": True
    }

agent = Agent(
    name="math checker",
    instructions="""
        Evaluate the given math expression.
        Return the result, whether it is even or not, and the number of operators used.
    """,
    model=global_model,
    output_type=MathOutput
)

result = Runner.run_sync(
    starting_agent=agent,
    input="what is 2+2",
)

print(result.final_output)