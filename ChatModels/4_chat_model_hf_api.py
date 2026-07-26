from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)


model = ChatHuggingFace(
    llm=llm,
    max_output_tokens=256,
    temperature=0.7
)

result = model.invoke("What is the capital of Bangladesh?")

print(result.content)
