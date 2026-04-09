from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough 


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
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short questions from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and Ques into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | llm | parser,
    'quiz':prompt2 | llm | parser
})

merged_chain = prompt3 | llm | parser

chain = parallel_chain | merged_chain

text = """
In this topic, we are not criticizing AUTOSAR, nor are we blindly supporting it. AUTOSAR is a well-designed and standardized framework that helps in building reliable and modular automotive software systems. It provides proper structure, task scheduling, and communication mechanisms required for real-time operations. However, in safety-critical systems like automotive braking, even small delays can become serious issues. These delays are not caused by AUTOSAR itself, but by how the system is designed and configured within it—such as poor scheduling, system overload, communication latency, or resource conflicts. AUTOSAR introduces multiple layers, which add slight overhead, and if not managed carefully, these can lead to deadline misses. So, the main focus of this case study is to analyze how timing issues can occur in AUTOSAR-based systems and how they can be optimized or improved to ensure safety and reliability.
"""

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()