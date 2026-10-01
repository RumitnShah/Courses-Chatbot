import os
import re
import random
import math
import logging
from collections import Counter

import redis
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file (locally) and Streamlit secrets (on Streamlit Cloud)
load_dotenv(override=True)


def get_secret(name, default=None):
    """Read a setting from the environment, falling back to Streamlit secrets."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        # Only touch st.secrets when a secrets.toml exists; otherwise Streamlit
        # shows a "No secrets found" error box on the page.
        if st.secrets.load_if_toml_exists() and name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return default


# Embedding model used to build the Pinecone index (1024 dimensions).
# It MUST stay the same as the model used in app_croq.py, otherwise the
# query vectors will not match the stored vectors.
EMBEDDING_MODEL = "intfloat/e5-large-v2"
INDEX_NAME = "course-database"
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-3.6-flash")


# ---------------------------------------------------------------------------
# Heavy resources are created ONCE per server process with st.cache_resource.
# Without this, Streamlit re-runs the whole script on every click (including
# moving the rating slider) and reloads the ~1.3 GB embedding model each time,
# which exhausts memory and crashes the app on Streamlit Cloud.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner=False)
def load_index():
    pinecone_api_key = get_secret("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file or Streamlit secrets")
    return Pinecone(api_key=pinecone_api_key).Index(INDEX_NAME)


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    return PineconeVectorStore(index=load_index(), embedding=load_embeddings(), text_key="text")


@st.cache_resource(show_spinner=False)
def load_llm():
    gemini_api_key = get_secret("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file or Streamlit secrets")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.3,
        api_key=gemini_api_key,
    )


@st.cache_resource(show_spinner=False)
def load_redis():
    return redis.Redis(
        host=get_secret("REDIS_HOST"),
        port=int(get_secret("REDIS_PORT", 6379)),
        password=get_secret("REDIS_PASSWORD"),
        decode_responses=True,
    )


vectorstore = load_vectorstore()
llm = load_llm()
redis_client = load_redis()

# ---------------------------------------------------------------------------
# Hybrid retrieval = embedding search + keyword (BM25) search.
#
# Embedding search alone ranks the right chunk poorly for questions like
# "semester 1" because the syllabus PDFs say "Semester I" / "Sem I" / "2nd
# Semester", and the extracted PDF text contains stray symbols. The keyword
# search normalises all of these to tokens like "semester_1", so the exact
# semester table is found reliably. Both result lists are merged with
# Reciprocal Rank Fusion.
# ---------------------------------------------------------------------------
ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8"}
ORDINAL_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
                 "sixth": "6", "seventh": "7", "eighth": "8", "final": "8"}
STOPWORDS = {"what", "are", "is", "the", "of", "in", "for", "a", "an", "and", "all", "to", "me",
             "give", "list", "show", "tell", "about", "which", "there", "any", "provide", "details", "b", "tech"}


def _number(token):
    """'iv' -> '4', 'fourth' -> '4', '4th' -> '4', '4' -> '4', otherwise None."""
    token = ROMAN.get(token, ORDINAL_WORDS.get(token, token))
    token = re.sub(r"^(\d+)(st|nd|rd|th)$", r"\1", token)
    return token if token.isdigit() and len(token) == 1 else None


def tokenize(text):
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    tokens = []
    for i, word in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else ""
        prev = words[i - 1] if i > 0 else ""
        # "semester 1", "Semester I", "Sem IV", "year 2"
        if word in ("semester", "sem", "year") and _number(nxt):
            tokens.append(("year_" if word == "year" else "semester_") + _number(nxt))
        # "first year", "2nd semester", "fourth sem"
        if nxt in ("semester", "sem", "year") and _number(word) and prev not in ("semester", "sem", "year"):
            tokens.append(("year_" if nxt == "year" else "semester_") + _number(word))
        if word not in STOPWORDS:
            tokens.append(word)
    return tokens


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1
        self.k1, self.b = k1, b
        df = Counter()
        for counts in self.docs:
            df.update(counts.keys())
        n = len(texts)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def top(self, query, k):
        q_tokens = tokenize(query)
        # "first year" in a question also means semesters 1 and 2, etc.
        for t in list(q_tokens):
            if t.startswith("year_"):
                year = int(t[5:])
                q_tokens += [f"semester_{2 * year - 1}", f"semester_{2 * year}"]
        scores = []
        for i, (counts, length) in enumerate(zip(self.docs, self.lengths)):
            score = 0.0
            for w in q_tokens:
                f = counts.get(w)
                if f:
                    weight = 3.0 if "_" in w else 1.0  # semester/year matches matter most
                    score += weight * self.idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        return [i for _, i in scores[:k]]


@st.cache_resource(show_spinner="Loading course data...", ttl=6 * 3600)
def load_keyword_index():
    """Load every chunk stored in Pinecone once and build a keyword index over it."""
    index = load_index()
    ids = []
    for page in index.list():
        ids.extend(page)
    docs = []
    for start in range(0, len(ids), 100):
        fetched = index.fetch(ids=ids[start:start + 100])
        for vid, vec in fetched.vectors.items():
            metadata = dict(vec.metadata or {})
            text = metadata.pop("text", "")
            docs.append(Document(id=vid, page_content=text, metadata=metadata))

    # Work out which chunk follows which. Chunks were split with a 150-char
    # overlap, so the start of chunk N+1 appears at the end of chunk N. This
    # lets us add the continuation of a table that was cut across two chunks.
    heads = {}
    for j, d in enumerate(docs):
        head = d.page_content.strip()[:60]
        if len(head) >= 40:
            heads.setdefault(head, []).append(j)
    next_chunk = {}
    for i, d in enumerate(docs):
        tail = d.page_content[-400:]
        for head, js in heads.items():
            if head in tail:
                for j in js:
                    if j != i:
                        next_chunk[d.id] = docs[j]
    return docs, BM25([d.page_content for d in docs]), next_chunk


def retrieve(question, k=12, candidates=30):
    """Merge embedding and keyword results with Reciprocal Rank Fusion."""
    dense_docs = vectorstore.similarity_search(question, k=candidates)
    kw_docs, bm25, next_chunk = load_keyword_index()
    keyword_docs = [kw_docs[i] for i in bm25.top(question, candidates)]

    scores, by_key = {}, {}
    for weight, ranked in ((1.0, dense_docs), (1.2, keyword_docs)):
        for rank, doc in enumerate(ranked):
            key = doc.id or doc.page_content[:200]
            by_key.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + weight / (60 + rank)
    fused = [by_key[key] for key in sorted(scores, key=scores.get, reverse=True)[:k]]

    # Keyword hits are the most precise signal for semester tables, so the top
    # keyword matches are always kept, followed by the fused results.
    best = keyword_docs[:6] + fused

    # Add the continuation of the top chunks (e.g. the rest of a semester table)
    results, seen = [], set()
    for rank, doc in enumerate(best):
        for d in (doc, next_chunk.get(doc.id) if rank < 10 else None):
            if d is not None and (d.id or d.page_content[:200]) not in seen:
                seen.add(d.id or d.page_content[:200])
                results.append(d)
    return results[:20]

# Define the prompt template
prompt_template = PromptTemplate(
    template="""You are an expert academic advisor specializing in curriculum information.

    Your task is to find and present semester-specific course information.

    Context:
    {context}

    Question:
    {question}

    Notes about the context:
    - It is text extracted from syllabus PDFs, so it may contain stray symbols such as ! ' " $ between words. Ignore them.
    - Semesters may be written as Roman numerals or ordinals: "Semester I" / "Sem I" / "1st Semester" all mean semester 1, "Semester IV" means semester 4, and so on.
    - "First year" means semesters 1 and 2, "second year" means semesters 3 and 4, etc.
    - "Computer Engineering", "CSE", "CE" and "Computer Science & Engineering" refer to the same program.

    Follow these steps:
    1. First, identify the specific semester and program mentioned in the question.
    2. Search the context for a match of that semester and program (using the notes above).
    3. If found, list all courses for that specific semester.
    4. If not found, clearly state that the information was not found in the provided context.
    5. Include course codes and names exactly as they appear.
    6. Format the response in an easy-to-read manner.

    Helpful Answer:""",
    input_variables=["context", "question"]
)


def run_qa_chain(question):
    """Retrieve relevant chunks and ask the LLM to answer from them."""
    docs = retrieve(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = prompt_template.format(context=context, question=question)
    response = llm.invoke(prompt)
    return {
        "result": response.content if hasattr(response, "content") else str(response),
        "source_documents": docs,
    }


def clean_answer(answer):
    # Remove the thinking part (anything between <think> and </think>)
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
    return (
        answer.replace("{", "")
        .replace("}", "")
        .replace('"', "")
        .replace("\\n", "\n")
        .strip()
    )


def get_ip():
    """Best-effort client IP (falls back to a shared key when unavailable)."""
    try:
        headers = st.context.headers
        forwarded = headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return headers.get("X-Real-Ip") or "unknown"
    except Exception:
        return "unknown"


def check_rate_limit():
    """Stop the app if this user has hit the daily query limit."""
    MAX_REQUESTS_PER_DAY = 200
    rate_limit_key = f"rate_limit_{get_ip()}"
    request_count = redis_client.get(rate_limit_key)

    if request_count is None:
        redis_client.set(rate_limit_key, 1, ex=86400)  # 1 day expiry
    else:
        if int(request_count) >= MAX_REQUESTS_PER_DAY:
            st.error("🚨 You have exceeded the daily query limit. Please try again tomorrow.")
            st.stop()
        redis_client.incr(rate_limit_key)


def get_source_display(docs):
    """Most common (source, URL) among the top retrieved chunks, as a markdown link."""
    search_results = docs[:5]
    source_counts = Counter(
        (doc.metadata.get("source", "Unknown"), doc.metadata.get("source_url", "No URL"))
        for doc in search_results
    )
    if not source_counts:
        return "- No source available."
    (source, url), _ = source_counts.most_common(1)[0]
    return f"- Source PDF: [{source}]({url})"


# ----------------------------- Streamlit UI --------------------------------
st.title("PDEU Courses Chatbot 🤖")

description = """Crafted with care by [Rumit Shah](https://www.linkedin.com/in/rumit-shah-537076303?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app) 💙. Explore the magic on [Github](https://github.com/RumitnShah/Courses-Chatbot/tree/main)"""
st.markdown(description, unsafe_allow_html=True)

st.markdown("""
🚨 **Important Notice:**

For security and privacy, avoid using public WiFi while chatting with this bot.
- Public networks share IPs, affecting rate limits and data security.

**Recommendation**: Use a personal or secure mobile network for the best experience. 🔒✅
""", unsafe_allow_html=True)

# Session state: history of Q&A and the latest answer (kept across reruns)
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "current" not in st.session_state:
    st.session_state.current = None

with st.form("my_form"):
    questions = [
        "Formulate your own question below",
        "What are the courses in Computer Science and Engineering semester 1?",
        "Are there elective specialization in computer engineering?",
        "Total credits in the first year of Computer Science Engineering?",
        "Provide details for Engineering Metallurgy course",
        "What are all the courses for Mechanical Engineering semester 4?",
        "What are the details of Electronics Devices and Circuits course?"
    ]

    selected_question = st.selectbox("Select your question:", questions, label_visibility="visible")

    st.markdown("<p style='text-align: center; font-size: 13px'>OR</p>", unsafe_allow_html=True)

    custom_question = st.text_area(
        "Enter your query here:",
        placeholder="Enter your custom question here...",
    )
    st.caption("A custom question is used when 'Formulate your own question below' is selected.")
    submitted = st.form_submit_button("Submit")

with open("loading_messages.txt", "r", encoding="utf-8") as f:
    loading_messages = [line.strip() for line in f if line.strip()]

if submitted:
    if selected_question == "Formulate your own question below":
        query = custom_question.strip()
    else:
        query = selected_question

    if not query:
        st.error("Please either select a question or write your own")
    else:
        try:
            with st.spinner(text=random.choice(loading_messages)):
                check_rate_limit()

                redis_key = f"query:{query}"
                cached_answer = redis_client.get(redis_key)

                if cached_answer:
                    answer = cached_answer
                    docs = retrieve(query, k=5)
                else:
                    result = run_qa_chain(query)
                    answer = clean_answer(result["result"])
                    docs = result["source_documents"]

                source_display = get_source_display(docs)

            st.session_state.current = {
                "question": query,
                "answer": answer,
                "sources": source_display,
                "cached": bool(cached_answer),
            }
            st.session_state.qa_history.append(
                {"question": query, "answer": answer, "sources": source_display}
            )
        except Exception as e:
            logging.exception("Error while answering query")
            st.session_state.current = None
            if "429" in str(e) or "quota" in str(e).lower():
                st.error("⏳ The AI service is busy right now (free-tier limit reached). Please try again in a minute.")
            else:
                st.error("Sorry, there was an issue processing your query.")
                st.error(str(e))

# Show the latest answer outside the `if submitted` block so it survives the
# rerun triggered by the rating slider (otherwise rating never gets saved).
current = st.session_state.current
if current:
    st.write("Answer:")
    st.markdown(current["answer"])
    st.markdown(current["sources"], unsafe_allow_html=True)

    answer_rating = st.slider("**Rate the provided answer {1=Worst, 10=Excellent}**", 1, 10, 5)

    # Cache answer in Redis if rating is above 7
    if answer_rating > 7 and not current["cached"]:
        redis_client.set(f"query:{current['question']}", current["answer"], ex=604800)  # 7 days
        current["cached"] = True
        st.success("Thanks! This answer will be reused for the same question.")

    st.write("⚠️ **Note:** AI can sometimes provide wrong answers. Please verify from the provided source.")

if st.session_state.qa_history:
    st.write("### Previous Questions and Answers")
    for qa_pair in st.session_state.qa_history:
        st.write(f"🤖 **Question:** {qa_pair['question']}")
        st.write(f"✨ **Answer:** {qa_pair['answer']}")
        st.write(f"📚 {qa_pair['sources']}")
        st.markdown("-----------------------------")
else:
    st.write("##### No previous questions and answers to display.")
