from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(               #kwargs -> keyword arguments 
        temperature=0.5,
        max_new_tokens=500
    )
)

model = ChatHuggingFace(llm = llm)

messages = [
    SystemMessage(content = "You are a helpful assistant"),
    HumanMessage(content = "tell me about langchain")
]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))

print(messages)