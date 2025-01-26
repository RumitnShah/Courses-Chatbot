from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import streamlit as st

# print(st.secrets['PINECONE_API_KEY'])
# load_dotenv(dotenv_path=r"C:\Users\Administrator\OneDrive\Documents\GitHub\Courses-Chatbot\.env")

# if "PINECONE_API_KEY" not in os.environ:
#     raise ValueError("Please set PINECONE_API_KEY in your .env file")
# Initialize Pinecone
pc = Pinecone(
    api_key=st.secrets['PINECONE_API_KEY']
)
index = pc.Index("courses-vector-database")

# Add embeddings
embeddings = OpenAIEmbeddings()

# Create vectorstore with embeddings
vectorstore = PineconeVectorStore(
    index=index, 
    embedding=embeddings,
    text_key="text"
)

# Create retriever with search parameters
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximal Marginal Relevance
    search_kwargs={
        "k": 50, # Number of documents to fetch
        "fetch_k": 40, # Fetch more candidates to allow better re-ranking
        "lambda_mult": 0.7  # Balance between relevance (1.0) and diversity (0.0)  # Fetch more relevant documents
    }
)
# Create QA chain with more detailed parameters
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        model="gpt-4",
        temperature=0.2,  # Slight creativity while maintaining accuracy
        max_tokens=300    # Allow longer responses
    ),
    chain_type="stuff",
    retriever=retriever,
    # return_source_documents=True  # Shows source of information
)

# Example query
query = "all courses in mechanical semester 1?"
result = qa.invoke(query)
print(result)