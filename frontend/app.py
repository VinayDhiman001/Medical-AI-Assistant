
import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="centered"
)

# Title
st.title("🏥 Medical AI Assistant")
st.markdown("Ask any medical question and get AI-powered answers")

# Sidebar
st.sidebar.title("Settings")
use_rag = st.sidebar.checkbox(
    "Use RAG (Document Based)", 
    value=False,
    help="Enable to answer from uploaded documents"
)

# Main input
question = st.text_input(
    "Enter your medical question:",
    placeholder="e.g. What is the treatment for Type 1 Diabetes?"
)

# Submit button
if st.button("Get Answer", type="primary"):
    if question:
        with st.spinner("Thinking..."):
            # Call our FastAPI
            payload = {
                "question": question,
                "use_rag": use_rag
            }
            response = requests.post(
                "http://localhost:8000/ask",
                json=payload
            )
            result = response.json()
        
        # Display answer
        st.success("Answer:")
        st.write(result["answer"])
        st.caption(f"Mode: {result['mode']}")
    else:
        st.warning("Please enter a question!")
