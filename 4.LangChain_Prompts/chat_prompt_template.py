from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate([

    # this is the way we will provide the prompt
    ('system', 'You are helpful {domain} expert'),
    ('human', 'Explain me in simple terms, what is {topic}'),

    # we are not going to use this like PromptMessage 
    # SystemMessage(content='You are helpful {domain} expert'),                      
    # HumanMessage(content='Explain me in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'cricket', 'topic':'dusra'})

print(prompt)