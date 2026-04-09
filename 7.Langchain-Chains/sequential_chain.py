from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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


prompt1 = PromptTemplate(
    template="Generate a detailed report about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template='Generate a 5 point summary on {text}',
    input_variables=['text']
)

parser=StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser

result = chain.invoke({'topic': 'Maharaja ranjit singh'})

print(result)

chain.get_graph().print_ascii()