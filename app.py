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

load_dotenv()

st.set_page_config(
    page_title="EZ AI-Assistant(made by Shashwat)",
    
    layout="wide"
)

# Initialize session state
for key in ["vectorstore", "retrieval_tool", "document_summary", "challenge_questions", "evaluation_result", "current_document"]:
    if key not in st.session_state:
        st.session_state[key] = None

def main():
    st.title("EZ AI-Assistant(made by Shashwat)")
    st.markdown("Upload a document (PDF or TXT) to activate: Auto-Summary, Ask Anything, Challenge Me.")

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Google API key missing. Add GOOGLE_API_KEY to .env.")
        st.stop()

    with st.sidebar:
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
        
        if uploaded_file:
            # Show current document info
            if st.session_state.current_document != uploaded_file.name:
                st.info(f"New document: {uploaded_file.name}")
                
            if st.button("Process Document"):
                process_uploaded_file(uploaded_file)

    # Display current document status
    if st.session_state.current_document:
        st.sidebar.success(f"Current document: {st.session_state.current_document}")
    
    if st.session_state.vectorstore is None:
        st.info(" Please upload and click process a document to begin.")
    else:
        tab1, tab2, tab3 = st.tabs([" Auto-Summary", " Ask Anything", " Challenge Me"])
        with tab1:
            display_auto_summary()
        with tab2:
            display_ask_anything()
        with tab3:
            display_challenge_me()

def process_uploaded_file(uploaded_file):
    with st.spinner("Processing document..."):
        # Clear previous session state
        st.session_state.document_summary = None
        st.session_state.challenge_questions = None
        st.session_state.evaluation_result = None
        
        suffix = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            processor = DocumentProcessor()
            file_type = uploaded_file.name.split(".")[-1].lower()
            
            # Process document (this will clear existing data automatically)
            vs = processor.process_document(tmp_path, file_type)
            
            # Update session state
            st.session_state.vectorstore = vs
            st.session_state.retrieval_tool = DocumentRetriever(vs)
            st.session_state.current_document = uploaded_file.name
            
            st.success(" Document processed successfully!")
            
        except Exception as e:
            st.error(f"Processing error: {e}")
        finally:
            os.unlink(tmp_path)

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1
    )

def display_auto_summary():
    st.header("Document Summary")
    st.write("Generate a  summary  based on the file given (max 150 words).")
    
    if st.session_state.current_document:
        st.info(f"Document: {st.session_state.current_document}")
    
    if st.button("Generate Summary", key="gen_summary"):
        with st.spinner("Generating summary..."):
            # Get document content using retrieval
            doc_content = st.session_state.retrieval_tool.retrieve("main topics and key points")
            
            # Create LangChain summarization chain
            prompt = ChatPromptTemplate.from_template("""
            Summarize the main points of the following document content in under 150 words.
            NOTE:(Focus on the key findings, important conclusions, and main topics.
            Provide specific references(like page no., article no. paragraph etc.) to the content when possible.)
            
            Document Content:
            {context}
            
            Summary (max 150 words):
            """)
            
            llm = get_llm()
            
            # Convert retrieved content to Document objects for the chain
            docs = [Document(page_content=doc_content)]
            
            # Create and run the chain
            chain = create_stuff_documents_chain(llm, prompt)
            summary = chain.invoke({"context": docs})
            
            st.session_state.document_summary = summary
            
    if st.session_state.document_summary:
        st.success("Summary Generated:")
        st.write(st.session_state.document_summary)

def display_ask_anything():
    st.header("Ask Anything")
    
    if st.session_state.current_document:
        st.info(f"Document: {st.session_state.current_document}")
    
    question = st.text_input("Your question:")
    
    if st.button("Get Answer", key="get_answer") and question:
        with st.spinner("Searching for answer..."):
            # Create QA chain using LangChain
            prompt = ChatPromptTemplate.from_template("""
            Answer the following question based solely on the provided context.
            If the answer is not in the context, clearly state that the information is not available in the document.
            Always cite specific parts of the document when providing answers.
            
            Question: {input}
            Context: {context}
            
            Answer:
            """)
            
            llm = get_llm()
            
            # Create retrieval chain
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Get answer
            response = retrieval_chain.invoke({"input": question})
            answer = response["answer"]
            
            st.info("Answer:")
            st.write(answer)

def display_challenge_me():
    st.header(" Challenge Me")
    
    if st.session_state.current_document:
        st.info(f"Document: {st.session_state.current_document}")
    
    if st.button("Generate Challenge Questions", key="gen_challenges"):
        with st.spinner("Creating challenges..."):
            # Get document content for question generation
            doc_content = st.session_state.retrieval_tool.retrieve("main topics concepts key information")
            
            prompt = ChatPromptTemplate.from_template("""
            Based on the following document content, generate exactly three challenging questions that test deep comprehension and logical reasoning.
            
            Document Content:
            {context}
            
            Requirements:
            - Generate exactly 3 questions
            - Each question must be answerable using only the document content
            - Focus on comprehension, analysis, and logical reasoning
            - Questions should test understanding of key concepts
            - Format each question clearly and number them
            
            Questions:
            """)
            
            llm = get_llm()
            
            # Convert retrieved content to Document objects
            docs = [Document(page_content=doc_content)]
            
            # Create and run the chain
            chain = create_stuff_documents_chain(llm, prompt)
            questions_text = chain.invoke({"context": docs})
            
            # Parse questions
            questions = [q.strip() for q in questions_text.split('\n') if q.strip() and ('?' in q or q.strip().endswith('.'))]
            
            if questions:
                st.session_state.challenge_questions = questions[:3]  # Ensure only 3 questions
                st.session_state.evaluation_result = None
            else:
                st.error("Could not generate questions from the document content.")

    if st.session_state.challenge_questions:
        st.subheader("Your Challenges:")
        
        with st.form("challenge_form"):
            answers = []
            for i, question in enumerate(st.session_state.challenge_questions):
                st.write(f"**Question {i+1}:** {question}")
                answer = st.text_area(f"Your answer:", key=f"ans_{i}", height=100)
                answers.append(answer)
            
            if st.form_submit_button("Submit Answers"):
                with st.spinner("Evaluating answers..."):
                    # Get document content for evaluation
                    doc_content = st.session_state.retrieval_tool.retrieve("all content for evaluation")
                    
                    prompt = ChatPromptTemplate.from_template("""
                    Evaluate the following answers based on the document content:
                    
                    Document Content:
                    {context}
                    
                    Questions and Answers:
                    {qa_pairs}
                    
                    For each answer, provide:
                    1. Correctness assessment (Correct/Partially Correct/Incorrect)
                    2. Detailed explanation with citations from the document
                    3. What the correct answer should be (if incorrect)
                    4. Specific references to document sections
                    
                    Evaluation:
                    """)
                    
                    # Format Q&A pairs
                    qa_pairs = ""
                    for i, (question, answer) in enumerate(zip(st.session_state.challenge_questions, answers)):
                        qa_pairs += f"\nQuestion {i+1}: {question}\nAnswer: {answer}\n"
                    
                    llm = get_llm()
                    
                    # Convert retrieved content to Document objects
                    docs = [Document(page_content=doc_content)]
                    
                    # Create and run evaluation chain
                    chain = create_stuff_documents_chain(llm, prompt)
                    evaluation = chain.invoke({"context": docs, "qa_pairs": qa_pairs})
                    
                    st.session_state.evaluation_result = evaluation
                    
        if st.session_state.evaluation_result:
            st.subheader("Evaluation Feedback:")
            st.write(st.session_state.evaluation_result)

if __name__ == "__main__":
    main()
