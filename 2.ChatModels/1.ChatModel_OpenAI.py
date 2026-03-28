from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=0, max_completion_tokens=10)   # here we can improve our code by providing more parameter one of the parameter is ` temperature `, its value lies between 0 to 2, it is a parameter that controls the randomness of a language model's output. it affects how creative or deterministic that responses are.   {https://miro.medium.com/v2/1*CqT1WTVIUsLlRQ61cQQX2A.png}, max completion tokens restrict the token while providing the response

result = model.invoke("What is the capital of India?")

print(result)   # so chatmodel will not give the plain text answer like llms, along with that it'll also provide some meta data like completion_tokens, audio_tokens, cached_tokens and many other meta-data,so generally what we do instead of calling result we try to fetch content inside the result by {`result.content`}

print(result.content)   #now this will give me simply answer
