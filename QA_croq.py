from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
import logging
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import streamlit as st
import re
import redis
import socket
import random
from collections import Counter

# Load environment variables from .env file
load_dotenv(override=True)

# Initialize Pinecone connection using API key
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

# Load Redis connection details from environment variables
host = os.environ.get("REDIS_HOST")
port = os.environ.get("REDIS_PORT")
password = os.environ.get("REDIS_PASSWORD")

# Initialize Redis client for caching and rate-limiting
redis_client = redis.Redis(
    host=host, port=port, password=password, decode_responses=True
)

# Connect to the Pinecone index
index = pc.Index("course-database")

# Initialize embeddings model for vector search
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

# Create a vector store instance using Pinecone
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Set up retriever with MMR (Maximal Marginal Relevance) search
retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={
        "k": 15,    # Number of documents to fetch
        "lambda_mult": 0.9  # Lower lambda_mult for more diverse results
    }
)

# Initialize LLM model using Groq API
llm = ChatGroq(
    model = "deepseek-r1-distill-llama-70b",   
    temperature = 0.3,  # Lower temperature for more deterministic answers
    api_key = os.environ['GROQ_API_KEY']
)

# Define the QA system using RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template="""You are an expert academic advisor specializing in curriculum information. 
                    Your task is to find and present semester-specific course information.
            input_variables=["context", "question"],
            Context: {context}

            Question: {question}

            Follow these steps:
            1. First, identify the specific semester and program mentioned in the question
            2. Search the context for an EXACT match of that semester and program
            3. If found, list all courses for that specific semester
            4. If not found display an error message
            5. Include course codes and names exactly as they appear
            6. Format the response in an easy-to-read manner

            Helpful Answer:"""
        ),
    },
)

# Set up the Streamlit UI
st.title("PDEU Courses Chatbot 🤖")

# Display author and GitHub link
description = """Crafted with care by [Rumit Shah](https://www.linkedin.com/in/rumit-shah-537076303?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app) 💙. Explore the magic on [Github](https://github.com/RumitnShah/Courses-Chatbot/tree/main)"""
st.markdown(description, unsafe_allow_html=True)

# Display important security notice
st.markdown("""
🚨 **Important Notice:**

For security and privacy, avoid using public WiFi while chatting with this bot. 
- Public networks share IPs, affecting rate limits and data security.

**Recommendation**: Use a personal or secure mobile network for the best experience. 🔒✅
""", unsafe_allow_html=True)

# Initialize session state for storing previous questions and answers
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# Function to get user's IP address
def get_ip():
    try:
        return socket.gethostbyname(socket.gethostname())  # Local network IP
    except:
        return "IP Address not available"


# Function to check rate limit per user
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

# Define the user input form
with st.form("my_form"):

    # Predefined questions for quick selection
    questions = [
        "Formulate your own question below",
        "What are the courses in Computer Science and Engineering semester 1?",
        "Are there elective specialization in computer engineering?",
        "Total credits in the first year of Computer Science Engineering?",
        "Provide details for Engineering Metallurgy course",
        "What are all the courses for Mechanical Engineering semester 4?",
        "What are the details of Electronics Devices and Circuits course?"
    ]
    
    selected_question = st.selectbox(
        "Select your question:",
        questions,
        label_visibility="visible"
    )

    # Allow user to enter custom question
    if selected_question == "Formulate your own question below":
        custom_question = st.text_area(
            "Enter your query here:",
            placeholder="Enter your custom question here..."
    )
    submitted = st.form_submit_button("Submit")

# Read funny loading messages from a file
with open("loading_messages.txt", "r") as f:
    loading_messages = f.readlines()

if submitted: 
    try: 
        loading_message = random.choice(loading_messages)
        # Adding loading message
        with st.spinner(text=loading_message):

            check_rate_limit()  # Check user's rate limit  
            if selected_question == "Formulate your own question below":
                if custom_question.strip():  # Check if custom question is not empty
                    query = custom_question
                else:
                    st.error("Please either select a question or write your own")
            else:
                query = selected_question

            # Check Redis cache for previous answer
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
        search_results = vectorstore.similarity_search(query, k=5)  # Get top 5 search results

        # Extract sources and count occurrences
        source_counts = Counter((doc.metadata.get('source', 'Unknown'), doc.metadata.get('source_url', 'No URL')) 
                        for doc in search_results)

        # Get the most common (source, URL) pair
        if source_counts:
            (most_common_source, most_common_url), _ = source_counts.most_common(1)[0]  # Unpacking the most frequent tuple

            # Markdown format for hyperlink
            source_display = f"- Source PDF: [{most_common_source}]({most_common_url})"
            st.markdown(source_display, unsafe_allow_html=True)  # Display hyperlink
        else:
            st.markdown("- No source available.")

        if query != "Formulate your own question below":
            st.session_state.qa_history.append({"question": query, "answer": answer, "sources": source_display})

        # User rates the answer
        answer_ratings = st.slider("**Rate the provided answer {1=Worst, 10=Excellent}**", 1, 10)

        # Cache answer in Redis if rating is above 7
        if answer_ratings > 7 and not cached_answer:
            redis_client.set(redis_key, answer, ex=604800)  # Cache for 7 days

        st.write("⚠️ **Note:** AI can sometimes provide wrong answers. Please verify from the provided source.")
            
    except Exception as e:
        logging.error(f"Embedding generation error: {e}")
        st.error("Sorry, there was an issue processing your query.")

if st.session_state.qa_history:
    # Display all previous questions and answers
    st.write("### Previous Questions and Answers")
    for qa_pair in st.session_state.qa_history:
        st.write(f"🤖 **Question:** {qa_pair['question']}")
        st.write(f"✨ **Answer:** {qa_pair['answer']}")
        st.write(f"📚 {qa_pair['sources']}")
        st.markdown("-----------------------------")

else:
    st.write("##### No previous questions and answers to display.")
