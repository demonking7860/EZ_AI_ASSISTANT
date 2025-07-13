# EZ AI-Assistant
NOTE:- I tried my best , but due to shortage of time i could not deploy on backend so i used Render and it has computing issue .
To show that i am capable and eligible for this internship role i am adding link of my 2 projects which showcase my excellence and knowledge . Rest is up to you .

Project 1- https://www.pandainuniv.com/
Project 2-https://github.com/demonking7860/Resume-Optimization-AI-Assistant , https://resume-optimization-ai-assistant-1.onrender.com/

A smart, document-aware assistant for research summarization, free-form question answering, and logic-based question generation and evaluation.


## Overview
EZ AI-Assistant enables users to upload a PDF or TXT document, then:
- **Auto Summary**: Generates a concise (≤150 words) summary.  
- **Ask Anything**: Answers free-form questions grounded in the document.  
- **Challenge Me**: Produces three logic-focused questions and evaluates user answers with detailed, citation-based feedback.  

## Features
- **Document Upload** (PDF/TXT)  
- **Context-Aware Summarization**  
- **Cited Question Answering**  
- **Logic-Based Question Generation & Evaluation**  
- **Ephemeral File Handling**: Processes uploads in temp storage, persists embeddings in ChromaDB.  

## Architecture  
![Architecture Diagram](./assets/architecture.pnglit app (`app.py`)  
2. **Document Processor**: Splits, embeds, and persists chunks via ChromaDB  
3. **Retrieval Tool**: Retrieves top-k chunks on demand  
4. **LLM**: Google Gemini via LangChain prompts  
5. **Render**: Hosts the Streamlit service with ephemeral upload handling and persistent vector store  

## Repository Structure
```
EZ_AI-ASSISTANT/
│
├── api/                        # (Optional) Serverless functions if refactored
├── assets/                     # Images and diagrams
│   └── architecture.png
├── db/                         # Local ChromaDB persistence
│   └── chroma_db/
├── tools/                      # Retrieval helper
│   └── retrieval_tool.py
├── utils/                      # Document processing module
│   └── document_processor.py
├── app.py                      # Streamlit entrypoint
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── README.md                   # This file
```

## Setup & Deployment

### Local Development
1. **Clone Repository**  
   ```bash
   git clone https://github.com/demonking7860/EZ_AI_ASSISTANT.git
   cd EZ_AI_ASSISTANT
   ```
2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Env Variables**  
   ```bash
   cp .env.example .env
   # Edit .env to add your GOOGLE_API_KEY
   ```
4. **Run Streamlit App**  
   ```bash
   streamlit run app.py
   ```
   Access at `http://localhost:8501`.




## Usage
1. **Upload Document**: Click “Choose a PDF or TXT file.”  
2. **Process Document**: Click “Process Document” in sidebar.  
3. **Auto-Summary**: Switch to the “Auto-Summary” tab and generate.  
4. **Ask Anything**: Enter a question in “Ask Anything” tab.  
5. **Challenge Me**: Generate and answer three logic questions under “Challenge Me.”

## Environment Variables
- `GOOGLE_API_KEY`: API key for Google Generative AI.  


## License
This project is licensed under the MIT License.

