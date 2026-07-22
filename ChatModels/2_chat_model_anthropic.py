from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0
)

result = model.invoke("Hi, Tell me about your self")

print(result.content)
