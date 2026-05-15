from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('poem.pdf')

docs = loader.load()

# print(len(docs))

print(docs[2].page_content)
print(docs[1].metadata)
