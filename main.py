from retrive_docs import VectorDB
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import pinecone
from langchain import LLMChain
from constants import CHAT_PROMPT, PROMPT
from langchain.llms import OpenAI

with open('openai_key.txt', 'r') as f:
    openai_key = f.read()

os.environ["OPENAI_API_KEY"] = openai_key

with open('pinecone-key.txt', 'r') as f:      
    api_key = f.read()      

pinecone.init(      
	api_key=api_key,      
	environment='gcp-starter'      
)  


vector = VectorDB(embeddings=OpenAIEmbeddings(),index_name='mining')

# llm = OpenAI()
llm = ChatOpenAI()

question = "what is the procedure for obtaining prospecting licenses or mining leases in respect of land in which the minerals vest in the government"

chain= LLMChain(llm=llm, prompt=CHAT_PROMPT, verbose=True)
print(chain.run({'question':question, 'context':vector._run(question)}))