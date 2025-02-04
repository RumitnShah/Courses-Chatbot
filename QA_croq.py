from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
import streamlit as st
import logging
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# Initialize Pinecone connection
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index = pc.Index("courses-db")

# Initialize embeddings
embeddings = HuggingFaceEmbeddings()

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
        "k": 50,    # Number of documents to retrieve
        "fetch_k": 40,      # Number of documents to return
        "lambda_mult": 0.7      # Lambda multiplier for MMR
    }
)

# Initialize Cohere client for generation
llm = ChatGroq(
    model = "deepseek-r1-distill-llama-70b",
    temperature = 0.7,
    api_key = os.environ['GROQ_API_KEY']
)

# Create QA chain with more detailed parameters
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template="""You are a helpful AI assistant for PDEU course information. 
            Always respond in a english, clear, concise, and friendly manner. 
            If the information is not in the context, say "I don't have enough information to answer that."

            Context: {context}

            Question: {question}
            Helpful Answer:""",
            input_variables=["context", "question"]
        ),
    # return_source_documents=True  # Shows source of information
    }
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
        placeholder = "E.g What are the courses in semester 1 of computer engineering?",
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

    answer = (
        answer.replace('{', '') # Remove JSON formatting characters
        .replace('}', '')   
        .replace('"', '')      
        .replace('\\n', '\n')  # Convert literal \n to actual newlines
        .strip()
    )

    if submitted:    
        # Display the clean answer
        if answer:
            st.write("Answer:")
            # Using markdown to properly render newlines
            st.markdown(answer)
            st.write(" Note: Sometimes LLM can Hallucinate the answer.")
        else:
            st.warning("No answer found for your question.")
                
except Exception as e:
    logging.error(f"Embedding generation error: {e}")
    st.error("Sorry, there was an issue processing your query.")
