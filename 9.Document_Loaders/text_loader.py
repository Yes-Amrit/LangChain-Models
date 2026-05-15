from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=google_api_key
)

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

prompt = PromptTemplate(
    template='Write a short summary about this content {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

chain = prompt | llm | parser
print(type(docs))
print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)

print(chain.invoke({'poem': docs[0].page_content}))