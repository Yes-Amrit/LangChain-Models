import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyCzu9nyTUOrinvkAr5VXXXXXXWUPNFjh5c"

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent 


search_tool = DuckDuckGoSearchRun()
tools = [search_tool]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

agent_executor = create_agent(
    model=llm, 
    tools=tools,
    system_prompt="You are a helpful assistant. Use your search tool to find up-to-date facts."
)
query = "fastest stumping in cricket in the world"
response = agent_executor.invoke({"messages": [("user", query)]})

print(response["messages"][-1].content)
