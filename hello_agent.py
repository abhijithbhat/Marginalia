import os
from pathlib import Path
from dotenv import load_dotenv
from strands import Agent
from strands.models.openai import OpenAIModel

# Load GROQ_API_KEY from .env
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)
groq_api_key = os.getenv("GROQ_API_KEY")

# Build Strands Agent using OpenAIModel provider configured for Groq
model = OpenAIModel(
    client_args={
        "api_key": groq_api_key,
        "base_url": "https://api.groq.com/openai/v1",
    },
    model_id="openai/gpt-oss-120b",
)

agent = Agent(model=model)

# Run agent once with the prompt
response = agent("In one sentence, what is the Model Context Protocol?")

# Print the full response
print(response)
