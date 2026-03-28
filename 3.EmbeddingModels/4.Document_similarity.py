from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Virat Kohli is known for his aggressive batting and consistency across all formats."
    "MS Dhoni is famous for his calm leadership and exceptional finishing skills."
    "Sachin Tendulkar is regarded as one of the greatest batsmen in cricket history."
    "Rohit Sharma is admired for his elegant stroke play and solid technique."
    "Ravindra Jadeja is known for his all-round performance and match-winning abilities."
]

query = 'tell me about virat kohli'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_documents(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]                # always pass 2D vector, we want simple list so we put [0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Similarity score is: ", score)