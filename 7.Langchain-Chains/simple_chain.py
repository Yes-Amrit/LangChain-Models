from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load env
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    print("No API key found")
    exit()

# 1. Define LLM (Gemini)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=google_api_key
)

# 2. Prompt
prompt = PromptTemplate(
    template="Write a short note about {topic}",
    input_variables=["topic"]
)

# 3. Parser
parser = StrOutputParser()

# 4. Chain
chain = prompt | llm | parser

# 5. Run
result = chain.invoke({"topic": "Chhatrapati Shivaji Maharaj"})
print(result)
chain.get_graph().print_ascii()