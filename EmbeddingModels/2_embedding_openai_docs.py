from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()



embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=32
)

documents = [
    "Dhaka is the capital of Bangladesh",
    "The capital of Bangladesh is Dhaka",
    "Bangladesh is a country in South Asia"
]

result = embedding.embed_documents(documents)

print(str(result))
