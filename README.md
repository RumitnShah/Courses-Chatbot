# Courses Chatbot

This chatbot is designed to provide course-related information for PDEU students using a conversational AI interface. It leverages LangChain, Pinecone, Redis, and Google Gemini to retrieve and generate responses based on stored course syllabi.

## Features
- Embeds and stores course syllabus PDFs in Pinecone for efficient retrieval.
- Uses hybrid search (semantic embeddings + keyword search) to fetch relevant course details, so questions like "semester 1" match "Semester I" in the syllabus PDFs.
- Provides responses using a Google Gemini model (default ```gemini-3.6-flash```, configurable with ```GEMINI_MODEL```).
- Implements Redis caching to store previous answers and limit query rates.
- Streamlit-based UI with a predefined set of questions and custom query input.
- Displays a clickable link to the syllabus PDF of the program asked about, for verification.
- Allows users to rate answers for quality improvement. 

## Prerequisites
- Python 3.10 or higher
- Make sure Python is installed on your machine. You can download it from python.org.
- Pinecone API Key
- Google Gemini API Key
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
- Run the following script to process and embed a course PDF (pass the PDF path):
```bash
python app_croq.py "Revised Syllabus/Mechanical.pdf"
```
- Create the account on Pinecone and get the API key.

- Set ```web_url``` and the ```source``` name in ```app_croq.py``` for each PDF before running it.
- Read [EMBEDDING_GUIDE.md](EMBEDDING_GUIDE.md) before creating new embeddings.
- Ensure metadata includes relevant details like source URL:
```bash
doc.metadata = {"source": pdf_path, "source_url": web_url}
```
2. Running the Chatbot Locally
- Launch the chatbot with Streamlit:
```bash
streamlit run QA_croq.py
```
- Get a Gemini API key from Google AI Studio.

- Create the account on Redis to store the user's data and get the API key. This will be used to store the results of the user's query and will not forward the same query to the Gemini API.

- Create the account on Streamlit to deploy the app. Store the API keys in the .env file as follows. Try editing the file ```.example.env``` to ```.env``` once the necessary information is stored.
- On Streamlit Cloud, add the same keys under **App settings → Secrets** (the ```.env``` file is not uploaded to GitHub).

## Purpose
The purpose of this chatbot is to provide a helpful resource for students and faculty seeking quick and accurate information about university courses. This chatbot was created with the intention of benefiting the academic community by making course details easily accessible.

We believe that finding syllabus information should be simple and efficient, and we hope this chatbot will help users retrieve semester-wise course details with ease. We strongly welcome any feedback or suggestions to improve this resource and make it even more useful for academic inquiries.

You can reach out to me via [Gmail](mailto:rumitshahn@gmail.com) or [LinkedIn](https://www.linkedin.com/in/rumit-shah-537076303). 

## Notes
- The bot combines embedding search and keyword (BM25) search, and adds the following chunk of each top match so semester tables split across chunks stay complete.
- The embedding model (```intfloat/e5-large-v2```, about 1.3 GB) loads once when the app starts, so the first question after a restart is slower.
- The Gemini free tier has small daily limits; the app shows a "try again later" message when the quota is used up.
- Redis ensures rate-limiting (200 queries per day per user IP) and caches responses for 7 days.
- Sources are cited in responses for transparency.

## Powered by
This example is powered by the following services:

- Hugging Face (Embedding Model)
- Google Gemini (AI API)
- Pinecone (Vector Database)
- Redis (Database)
- Streamlit (App Deployment)