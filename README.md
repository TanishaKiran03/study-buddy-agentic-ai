# Study Buddy - Agentic AI Physics Tutor

Study Buddy is an agentic AI-powered physics tutoring assistant for B.Tech students. It helps students understand core physics concepts using a knowledge base, retrieval, tools, memory, and a Streamlit chat interface.

The project was built as part of the Agentic AI Hands-On Course 2026 capstone project.

## Project Overview

Students often study physics at odd hours without access to tutors. Study Buddy provides a 24/7 assistant that can explain physics concepts, cite retrieved source topics, use tools for calculations and date/time queries, remember the student's name within a session, and admit when a question is outside its knowledge base.

## Features

- 12-document Physics knowledge base
- ChromaDB-based retrieval
- SentenceTransformer embeddings using `all-MiniLM-L6-v2`
- LangGraph StateGraph workflow
- Groq LLM integration
- MemorySaver with `thread_id` for session memory
- Router node for deciding between:
  - retrieval
  - tool use
  - memory-only response
- Calculator tool for numeric physics expressions
- Datetime tool for current date/time queries
- Faithfulness evaluation node
- Streamlit web chat interface
- Runtime Groq API key input, with no hardcoded API key

## Topics Covered

The knowledge base includes:

1. Newton's Laws of Motion
2. Work, Energy and Power
3. Gravitation
4. Thermodynamics
5. Waves and Oscillations
6. Optics - Reflection and Refraction
7. Electrostatics
8. Current Electricity
9. Magnetism and Electromagnetic Induction
10. Modern Physics - Photoelectric Effect and Atomic Models
11. Projectile Motion and Kinematics
12. Rotational Motion

## Project Structure

text
study-buddy-agentic-ai/
├── day13_capstone 2.ipynb
├── agent.py
├── capstone_streamlit.py
├── requirements.txt
└── .gitignore

## Files
day13_capstone 2.ipynb
Main capstone notebook containing:

problem statement
knowledge base setup
ChromaDB vector store
state design
node functions
LangGraph assembly
testing
evaluation
deployment notes
written summary
agent.py
Reusable backend implementation for Study Buddy. It contains the agent logic used by the Streamlit app.

capstone_streamlit.py
Streamlit web interface for chatting with Study Buddy.

requirements.txt
Python dependencies required to run the project.

## API Key Setup
This project uses Groq for LLM calls.

The Groq API key is not hardcoded in the code or notebook.

For Notebook
When running the notebook, it asks for the Groq API key at runtime:

Enter your Groq API key:
The key is stored only in the current notebook session.

For Streamlit App
The Streamlit app provides a password input field in the sidebar where the user can enter their Groq API key.

The key is used only for the running session and is not saved in the project files.

## Agent Workflow
The LangGraph workflow follows this structure:

User Question
    ↓
memory_node
    ↓
router_node
    ↓
retrieval_node / tool_node / skip_retrieval_node
    ↓
answer_node
    ↓
eval_node
    ↓
save_node
    ↓
END
