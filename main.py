import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Updated import: pulling the retriever from vector.py
from vector import retriever

# Configure the Streamlit page
st.set_page_config(page_title="Startup Intelligence Assistant", page_icon="🚀", layout="centered")
st.title("🚀 Startup Dataset RAG Chatbot")
st.write("Ask me anything about the startup dataset (funding, city, founders, relationships, milestones, etc.)")

# --- 1. Setup LLM & Chain (Cached) ---
@st.cache_resource
def get_chain():
    model = OllamaLLM(model="gemma3:1b")

    template = """
    You are a factual startup data assistant.

    You must strictly follow these rules:
    - Use ONLY the provided startup dataset records.
    - Do NOT hallucinate or assume missing values.
    - Do NOT generate predictions or business insights unless explicitly present in the records.
    - Do NOT compute derived metrics unless they are already present.
    - If the answer is not found in the records, respond exactly with:
      "The dataset does not contain this information."

    Startup dataset records:
    {records}

    User question:
    {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model

chain = get_chain()

# --- 2. Manage Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. Handle User Input ---
if question := st.chat_input("Ask a startup-related question (e.g., 'Which startups were founded in 2012?' )"):

    # Display the user's question
    with st.chat_message("user"):
        st.markdown(question)

    # Save the user's question to state
    st.session_state.messages.append({"role": "user", "content": question})

    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing startup dataset..."):

            # Retrieve relevant startup records using vector retriever
            records = retriever.invoke(question)

            # Invoke LLM with grounded context
            response = chain.invoke({
                "records": records,
                "question": question
            })

            st.markdown(response)

    # Save the assistant's response to state
    st.session_state.messages.append({"role": "assistant", "content": response})
