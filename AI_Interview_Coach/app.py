import streamlit as st
import re

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("🎯 Interview Settings")

candidate_name = st.sidebar.text_input(
    "👤 Candidate Name"
)

topic = st.sidebar.selectbox(
    "💻 Topic",
    ["Python", "SQL", "OOP"]
)

difficulty = st.sidebar.selectbox(
    "📈 Difficulty",
    ["Beginner", "Intermediate", "Advanced"]
)


# -----------------------------
# Main Page
# -----------------------------

st.title("🤖 AI Interview Coach")

st.write(
    "Practice your interview with AI — one question at a time 🎯"
)


# -----------------------------
# Load Interview Notes
# -----------------------------

loader = TextLoader(
    "documents/interview_notes.txt",
    encoding="utf-8"
)

documents = loader.load()


# -----------------------------
# Create Embeddings
# -----------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# -----------------------------
# Create FAISS Vector Database
# -----------------------------

vectorstore = FAISS.from_documents(
    documents,
    embeddings
)


# -----------------------------
# Connect to Ollama AI
# -----------------------------

llm = OllamaLLM(
    model="llama3.2"
)


# -----------------------------
# Session State
# -----------------------------

if "interview_question" not in st.session_state:
    st.session_state.interview_question = None

if "evaluation" not in st.session_state:
    st.session_state.evaluation = None

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "user_answer" not in st.session_state:
    st.session_state.user_answer = ""

if "score" not in st.session_state:
    st.session_state.score = None


# -----------------------------
# Start Interview
# -----------------------------

start_interview = st.button("🚀 Start Interview")


if start_interview:

    st.session_state.question_count += 1

    st.session_state.evaluation = None
    st.session_state.user_answer = ""
    st.session_state.score = None

    # Search relevant knowledge using RAG
    rag_results = vectorstore.similarity_search(
        f"{topic} {difficulty} interview question",
        k=3
    )

    # Combine retrieved knowledge
    context = "\n\n".join(
        result.page_content
        for result in rag_results
    )

    # Generate interview question
    interview_question = llm.invoke(
        f"""
You are an AI interview coach.

Create ONE {difficulty}-level {topic} interview question.

Use the following interview knowledge:

{context}

Ask only the interview question.
Do not give the answer.
Do not explain anything else.
"""
    )

    st.session_state.interview_question = interview_question


# -----------------------------
# Display Question
# -----------------------------

if st.session_state.interview_question:

    st.info(
        f"🎯 Interview Progress: Question {st.session_state.question_count}"
    )

    st.subheader("🤖 Interview Question")

    st.write(
        st.session_state.interview_question
    )


    # -----------------------------
    # Candidate Answer
    # -----------------------------

    user_answer = st.text_area(
        "✍️ Your Answer",
        key="user_answer"
    )


    # -----------------------------
    # Submit Answer
    # -----------------------------

    submit_answer = st.button(
        "📊 Submit Answer"
    )


    if submit_answer:

        evaluation = llm.invoke(
            f"""
You are an AI interview evaluator for {candidate_name}.

Interview Topic:
{topic}

Difficulty:
{difficulty}

Interview Question:
{st.session_state.interview_question}

Candidate Answer:
{user_answer}

Evaluate the candidate's answer.

Give:
1. Score out of 10
2. What is correct
3. What is missing or incorrect
4. One improvement suggestion

Keep the evaluation simple and easy to understand.
"""
        )

        st.session_state.evaluation = evaluation

        # Extract score
        score = re.search(
            r'(\d+)\s*(?:/10|out of 10)',
            evaluation,
            re.IGNORECASE
        )

        st.session_state.score = score


# -----------------------------
# Display Evaluation
# -----------------------------

if st.session_state.evaluation:

    st.subheader("📊 AI Evaluation")


    # Display Score
    if st.session_state.score:

        st.metric(
            "🏆 Score",
            f"{st.session_state.score.group(1)}/10"
        )


    # Display Full Evaluation
    st.write(
        st.session_state.evaluation
    )


    # -----------------------------
    # Follow-up Question
    # -----------------------------

    follow_up = st.button(
        "🔄 Ask Follow-up Question"
    )


    if follow_up:

        follow_up_question = llm.invoke(
            f"""
You are an AI interview coach.

Previous Interview Question:
{st.session_state.interview_question}

Candidate Answer:
{user_answer}

AI Evaluation:
{st.session_state.evaluation}

Based on the candidate's answer and evaluation,
ask ONE short follow-up interview question.

The follow-up question should test a concept
the candidate needs to improve.

Ask only the question.
Do not give the answer.
"""
        )

        st.subheader("🔄 Follow-up Question")

        st.write(
            follow_up_question
        )