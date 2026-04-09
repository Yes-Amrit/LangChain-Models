from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=google_api_key
)

# ✅ Pydantic model
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative']

parser2 = PydanticOutputParser(pydantic_object=Feedback)
parser = StrOutputParser()

# ✅ classifier prompt
prompt1 = PromptTemplate(
    template='Classify sentiment (positive/negative): {feedback}\n{format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# ✅ response prompts
prompt2 = PromptTemplate(
    template='Write a response to this positive feedback:\n{feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write a response to this negative feedback:\n{feedback}',
    input_variables=['feedback']
)

# ✅ FIX: Keep original input + sentiment together
chain = RunnablePassthrough.assign(
    sentiment=classifier_chain
) | RunnableBranch(
    (lambda x: x["sentiment"].sentiment == "positive", prompt2 | model | parser),
    (lambda x: x["sentiment"].sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not determine sentiment")
)

# Run
print(chain.invoke({'feedback': 'This is a beautiful phone'}))

# Graph
chain.get_graph().print_ascii()