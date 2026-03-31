#  Builtin Structured output is only supported in some of the llms like OpenAI, Models like Hugging Face TinyLlama don’t reliably follow structured schemas

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(               #kwargs -> keyword arguments 
        temperature=0.5,
        max_new_tokens=500
    )
)

model = ChatHuggingFace(llm=llm)

# this is specially for hugging face llms because they are not known to structured output, so we would follow other pattern and for OpenAI llms which is pre trained for structured output, we will choose 2nd method 

# 1st method 

prompt = """
Extract the following from the review:

1. Summary (1 line)
2. Sentiment (Positive/Negative/Neutral)

Return ONLY in this JSON format:
{
  "summary": "...",
  "sentiment": "..."
}

Review:
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
"""
# 2nd method 
# simple TypedDict
# class Review(TypedDict):
#    summary: str 
#    sentiment: str

# Annotated TypedDict
# class Review(TypedDict):
#   key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
#   summary: Annotated[str, "A brief summary of the review"]
#   sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
#   pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
#   cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
#   name: Annotated[Optional[str]], "Write the name of the reviewer"]


result = model.invoke(prompt)

print(result.content)