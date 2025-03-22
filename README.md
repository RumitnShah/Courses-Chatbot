# Courses Chatbot

This chatbot is designed to provide course-related information for PDEU students using a conversational AI interface. It leverages LangChain, Pinecone, Redis, and Groq's LLM to retrieve and generate responses based on stored course syllabi.

## Features
- Embeds and stores course syllabus PDFs in Pinecone for efficient retrieval.
- Uses MMR-based semantic search to fetch relevant course details.
- Provides responses using the ```deepseek-r1-distill-llama-70b``` model from Groq.
- Implements Redis caching to store previous answers and limit query rates.
- Streamlit-based UI with a predefined set of questions and custom query input.
- Displays source information with clickable link for verification.
- Allows users to rate answers for quality improvement. 

## Prerequisites
- Python 3.6 or higher
- Make sure Python is installed on your machine. You can download it from python.org.
- Pinecone API Key
- Groq API Key
- Redis Server (for caching and rate-limiting)

## Installation
1. Clone the given Repository

2. Set Up a Virtual Environment (recommended)
```bash
python -m venv venv
```
3. Activate Virtual Environment 
```bash
On Windows: venv\Scripts\activate
On macOS: source venv/bin/activate
```
4. Configure the Environment File
- Locate the ```.example.env``` file in the repository.
- Rename it to ```.env```.
- Add the required API keys obtained from the respective sources to ```.env``` file.

5. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
1. Embedding PDFs into Pinecone
- Run the following script to process and embed course PDFs:
```bash
python app_croq.py
```
- Create the account on Pinecone and get the API key.

- Add your pdf path in the given line
```bash
pdf_path = "pdfs/sample.pdf"
```
- Ensure metadata includes relevant details like source URL:
```bash
doc.metadata = {"source": pdf_path, "source_url": web_url}
```
2. Running the Chatbot Locally
- Launch the chatbot with Streamlit:
```bash
streamlit run chatbot.py
```
- Create the account on Croq and get the API key.

- Create the account on Redis to store the user's data and get the API key. This will be used to store the results of the user's query and will not forward the same query to the Croq API.

- Create the account on Streamlit to deploy the app. Store the API keys in the .env file as follows. Try editing the file ```.example.env``` to ```.env``` once the necessary information is stored.

## Purpose
The purpose of this chatbot is to provide a helpful resource for students and faculty seeking quick and accurate information about university courses. This chatbot was created with the intention of benefiting the academic community by making course details easily accessible.

We believe that finding syllabus information should be simple and efficient, and we hope this chatbot will help users retrieve semester-wise course details with ease. We strongly welcome any feedback or suggestions to improve this resource and make it even more useful for academic inquiries.

You can reach out to me via [Gmail](mailto:rumitshahn@gmail.com) or [LinkedIn](https://www.linkedin.com/in/rumit-shah-537076303). 

## Notes
- The bot supports Maximal Marginal Relevance (MMR) search with k=15 and lambda_mult=0.9 for diverse retrieval.
- Redis ensures rate-limiting (20 queries per day per user IP) and caches responses for 7 days.
- Sources are cited in responses for transparency.

## Powered by
This example is powered by the following services:

- Hugging Face (Embedding Model)
- Croq Cloud (AI API)
- Redis (Database)
- Streamlit (App Deployment)