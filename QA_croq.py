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
import re
import redis
import socket

load_dotenv()

# Initialize Pinecone connection
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

host = os.environ.get("REDIS_HOST")
port = os.environ.get("REDIS_PORT")
password = os.environ.get("REDIS_PASSWORD")

redis_client = redis.Redis(
    host=host, port=port, password=password, decode_responses=True
)

index = pc.Index("course-database")

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
        "k": 32,    # Number of documents to retrieve
        "fetch_k": 32,      # Number of documents to return
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
            template="""You are an expert assistant. Your task is to answer questions based solely on the context provided in the PDF sources. 

            STRICT RULES:
            1. Only use information directly from the context.
            2. If the information is not found, respond with: "I cannot find the specific information in the provided sources."
            3. Do NOT guess, infer, or fabricate any information.
            4. Always mention the section or page number where you found the information, if available.
            5. If there is any ambiguity, say so clearly.
            6. Only provide names of subjects, courses, or professors if they are explicitly mentioned in the context.
            7. Only provide information based on question asked.

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

# Important Notice
st.markdown("""
🚨 **Important Notice:**

For security and privacy, avoid using public WiFi while chatting with this bot. 
- Public networks share IPs, affecting rate limits and data security.

**Recommendation**: Use a personal or secure mobile network for the best experience. 🔒✅
""", unsafe_allow_html=True)

# Get user's IP address
def get_ip():
    try:
        return socket.gethostbyname(socket.gethostname())  # Local network IP
    except:
        return "unknown"


# Rate Limiting Per User (Based on IP Address)
def check_rate_limit():
    user_ip = get_ip()
    rate_limit_key = f"rate_limit_{user_ip}"
    request_count = redis_client.get(rate_limit_key)

    MAX_REQUESTS_PER_DAY = 200  # Set the daily query limit

    if request_count is None:
        redis_client.set(rate_limit_key, 1, ex=86400)  # 1 day expiry
    else:
        request_count = int(request_count)
        if request_count >= MAX_REQUESTS_PER_DAY:
            st.error("🚨 You have exceeded the daily query limit. Please try again tomorrow.")
            st.stop()
        redis_client.incr(rate_limit_key)

with st.form("my_form"):
    # text = st.text_area(
    #     "Enter your question related to syllabus:",
    #     placeholder = "E.g What are the courses in semester 1 of computer engineering?",
    # )
    # submitted = st.form_submit_button("Submit")

    questions = [
        "Write your own question below",
        "What are the courses in Computer Engineering semester 1?",
        "What are the courses in Mechanical semester 2?",
        "What is the duration of the Biotechnology?",
        "What are the courses for Information Technology?"
    ]
    
    selected_question = st.selectbox(
        "Select your question:",
        questions,
        label_visibility="collapsed"
    )
    # Text area for custom question
    if selected_question == "Write your own question below":
        custom_question = st.text_area(
            "Write your own question:",
            placeholder="Enter your custom question here..."
    )
    submitted = st.form_submit_button("Submit")

if submitted:  
    check_rate_limit()  # Check user's rate limit  
    if selected_question == "Write your own question below":
        if custom_question.strip():  # Check if custom question is not empty
            query = custom_question
        else:
            st.error("Please either select a question or write your own")
    else:
        query = selected_question

    try:
        # Check if the answer is already cached in Redis
        redis_key = f"query:{query}"
        cached_answer = redis_client.get(redis_key)
        
        if cached_answer:
            answer = cached_answer
        else:
            result = qa.invoke(query)

            # Extract and clean the answer
            if isinstance(result, dict):
                answer = result.get('result', '')
            elif isinstance(result, str):
                answer = result
            else:
                answer = str(result)

            # Remove the thinking part (anything between <think> and </think>)
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)

            answer = (
                answer.replace('{', '') # Remove JSON formatting characters
                .replace('}', '')   
                .replace('"', '')      
                .replace('\\n', '\n')  # Convert literal \n to actual newlines
                .strip()
            )

        # Display the clean answer
        if answer:
            st.write("Answer:")
            # Using markdown to properly render newlines
            st.markdown(answer)

        # Perform similarity search
        search_results = vectorstore.similarity_search(query, k=2)

        # Print sources for the query
        for i, doc in enumerate(search_results, 1):
            source_path = doc.metadata.get('source', 'Source not available')
            source_url = doc.metadata.get('source_url', 'source')
        
        if source_path:
            st.markdown(f"- Source PDF: [{source_path}]({source_url})")

        # User rates the answer
        answer_ratings = st.slider("**Rate the provided answer {1=Worst, 10=Excellent}**", 1, 10)

        # Cache answer in Redis if rating is above 7
        if answer_ratings > 7 and not cached_answer:
            redis_client.set(redis_key, answer, ex=604800)  # Cache for 7 days

        st.write("⚠️ **Note:** AI can sometimes provide wrong answers. Please verify from the provided source.")
            
    except Exception as e:
        logging.error(f"Embedding generation error: {e}")
        st.error("Sorry, there was an issue processing your query.")
