from langchain_pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone
import streamlit as st
import os
from dotenv import load_dotenv
import cohere
from langchain_cohere import CohereEmbeddings
from langchain.chains import RetrievalQA
# from langchain_community.chat_models import CohereChat
from langchain_community.llms import Cohere as CohereLLM 
import streamlit as st
import logging

load_dotenv()

# Initialize Pinecone connection
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index = pc.Index("courses-database")

# Initialize Cohere embeddings
# cohere_client = cohere.Client(api_key=os.environ['COHERE_API_KEY'])
embeddings = CohereEmbeddings(model="large")

# Create vectorstore instance for the existing index
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Set up retriever with MMR search
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 50,
        "fetch_k": 40,
        "lambda_mult": 0.7
    }
)

# # Get relevant documents
# docs = retriever.get_relevant_documents(question)

# # Combine retrieved documents into a single context
# context = " ".join([doc.page_content for doc in docs])

# # Output the retrieved context
# print(f"Context for the question '{question}':\n{context}")

# Initialize Cohere client for generation
cohere_client = cohere.Client(os.environ['COHERE_API_KEY'])
llm = CohereLLM(client=cohere_client, model="command-r-08-2024")

# Create QA chain with more detailed parameters
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    # return_source_documents=True  # Shows source of information
)

# # Define the question
# question = "which are the courses in cse semester 1?"
# result = qa.invoke(question)
# print(result)



st.title("PDEU Courses Chatbot 🤖")

description = """Crafted with care by [Rumit Shah](https://www.linkedin.com/in/rumit-shah-537076303?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app) 💙. Explore the magic on [Github](https://github.com/RumitnShah/Courses-Chatbot/tree/main)"""
st.markdown(description, unsafe_allow_html=True)

with st.form("my_form"):
    text = st.text_area(
        "Enter your question related to syllabus:",
        placeholder = "E.g What are the courses in semester 1 of mechanical engineering?",
    )
    submitted = st.form_submit_button("Submit")

query = text
try:
    result = qa.invoke(query)

    # Extract and clean the answer
    if isinstance(result, dict):
        answer = result.get('result', '')
    elif isinstance(result, str):
        answer = result
    else:
        answer = str(result)
        
        # Clean the answer:
        # 1. Remove JSON formatting characters
        # 2. Convert literal \n to actual newlines
        # 3. Remove any extra whitespace
    answer = (
        answer.replace('{', '')
        .replace('}', '')
        .replace('"', '')
        .replace('\\n', '\n')  # Convert \n to actual newlines
        .strip()
    )
        
        # Display the clean answer
    if answer:
        st.write("Answer:")
            # Using markdown to properly render newlines
        st.markdown(answer)
    else:
        st.warning("No answer found for your question.")
            
except Exception as e:
    logging.error(f"Embedding generation error: {e}")
    st.error("Sorry, there was an issue processing your query.")

# if submitted:
#     st.info(result.get('result', 'No result found'))