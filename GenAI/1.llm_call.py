from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
import os
load_dotenv()

llm = ChatVertexAI(
    model_name="gemini-2.5-pro",
    project = os.getenv("GCP_PROJECT_ID"),
    location = "us-central1",
    temperature=2,   # 0-2
    max_tokens = 2000

    )


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Provide me Jokes on AI?"}
]

print(llm.invoke(messages).content)