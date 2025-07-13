import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from utils.document_processor import DocumentProcessor
from tools.retrieval_tool import DocumentRetriever

# Load .env in local dev; in Render, use st.secrets or env vars directly
load_dotenv()

st.set_page_config(
    page_title="EZ AI-Assistant",
    layout="wide"
)

# Initialize session state
for key in ["vectorstore", "retrieval_tool", "document_summary", 
            "challenge_questions", "evaluation_result", "current_document"]:
    if key not in st.session_state:
        st.session_state[key] = None

@st.cache_resource(show_spinner=False)
def get_processor():
    return DocumentProcessor()

@st.cache_resource(show_spinner=False)
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.1
    )

def main():
    st.title("EZ AI-Assistant")
    st.markdown("Upload a PDF/TXT, then use Auto-Summary, Ask Anything, or Challenge Me.")

    if not (os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")):
        st.error("Google API key missing. Set `GOOGLE_API_KEY` as env var or secret.")
        st.stop()

    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader("Choose a PDF or TXT", type=["pdf", "txt"])
        if uploaded_file:
            if st.session_state.current_document != uploaded_file.name:
                st.info(f"New document: {uploaded_file.name}")
            if st.button("Process Document"):
                process_uploaded_file(uploaded_file)

        if st.session_state.current_document:
            st.sidebar.success(f"Current: {st.session_state.current_document}")
        else:
            st.info("Upload and click Process to begin.")

    if st.session_state.vectorstore:
        tab1, tab2, tab3 = st.tabs(["Auto-Summary", "Ask Anything", "Challenge Me"])
        with tab1:
            display_auto_summary()
        with tab2:
            display_ask_anything()
        with tab3:
            display_challenge_me()

def process_uploaded_file(uploaded_file):
    with st.spinner("Processing document..."):
        st.session_state.document_summary = None
        st.session_state.challenge_questions = None
        st.session_state.evaluation_result = None

        suffix = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            processor = get_processor()
            file_type = uploaded_file.name.split(".")[-1].lower()
            vs = processor.process_document(tmp_path, file_type)
            st.session_state.vectorstore = vs
            st.session_state.retrieval_tool = DocumentRetriever(vs)
            st.session_state.current_document = uploaded_file.name
            st.success("Document processed successfully!")
        except Exception as e:
            st.error(f"Processing error: {e}")
        finally:
            os.unlink(tmp_path)

def display_auto_summary():
    st.header("Document Summary")
    st.write("Generate a concise summary (≤150 words).")
    if st.button("Generate Summary", key="gen_summary"):
        if not st.session_state.retrieval_tool:
            st.error("No document processed.")
            return
        with st.spinner("Generating summary..."):
            doc_content = st.session_state.retrieval_tool.retrieve("main topics and key points")
            prompt = ChatPromptTemplate.from_template("""
Summarize the main points of the following document in under 150 words.
Document Content:
{context}
Summary:
""")
            llm = get_llm()
            docs = [Document(page_content=doc_content)]
            chain = create_stuff_documents_chain(llm, prompt)
            summary = chain.invoke({"context": docs})
            st.session_state.document_summary = summary
    if st.session_state.document_summary:
        st.success("Summary:")
        st.write(st.session_state.document_summary)

def display_ask_anything():
    st.header("Ask Anything")
    question = st.text_input("Your question:", key="ask_input")
    if st.button("Get Answer", key="get_answer") and question:
        with st.spinner("Searching for answer..."):
            prompt = ChatPromptTemplate.from_template("""
Answer the question based solely on the context. Cite where possible.
Question: {input}
Context: {context}
Answer:
""")
            llm = get_llm()
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({"input": question})
            answer = response["answer"]
            st.info("Answer:")
            st.write(answer)

def display_challenge_me():
    st.header("Challenge Me")
    if st.button("Generate Challenge Questions", key="gen_challenges"):
        with st.spinner("Creating challenges..."):
            doc_content = st.session_state.retrieval_tool.retrieve("key concepts and logic")
            prompt = ChatPromptTemplate.from_template("""
Based on the document content, generate exactly 3 comprehension and logic questions.
Document Content:
{context}
Questions:
""")
            llm = get_llm()
            docs = [Document(page_content=doc_content)]
            chain = create_stuff_documents_chain(llm, prompt)
            questions_text = chain.invoke({"context": docs})
            questions = [q.strip() for q in questions_text.split("\n") if q.strip()]
            st.session_state.challenge_questions = questions[:3]
            st.session_state.evaluation_result = None

    if st.session_state.challenge_questions:
        with st.form("challenge_form"):
            answers = []
            for i, question in enumerate(st.session_state.challenge_questions):
                st.write(f"**Q{i+1}:** {question}")
                ans = st.text_area("Your answer:", key=f"ans_{i}", height=100)
                answers.append(ans)
            if st.form_submit_button("Submit Answers"):
                with st.spinner("Evaluating answers..."):
                    qa_pairs = ""
                    for i, (q, a) in enumerate(zip(st.session_state.challenge_questions, answers)):
                        qa_pairs += f"Q{i+1}: {q}\nA: {a}\n"
                    doc_content = st.session_state.retrieval_tool.retrieve("full content")
                    prompt = ChatPromptTemplate.from_template("""
Evaluate the answers based on the document. Provide correctness, explanation, and citations.
Document Content:
{context}
Q&A:
{qa_pairs}
Evaluation:
""")
                    llm = get_llm()
                    docs = [Document(page_content=doc_content)]
                    chain = create_stuff_documents_chain(llm, prompt)
                    evaluation = chain.invoke({"context": docs, "qa_pairs": qa_pairs})
                    st.session_state.evaluation_result = evaluation

    if st.session_state.evaluation_result:
        st.subheader("Evaluation Feedback:")
        st.write(st.session_state.evaluation_result)

if __name__ == "__main__":
    main()
