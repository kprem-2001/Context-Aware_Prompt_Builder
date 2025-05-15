import streamlit as st
import os
import openai
from openai import OpenAI
from logger_config import logging
from langchain_pinecone import PineconeVectorStore
from embedding import embedding

# Setup
st.set_page_config(page_title="🧠 Context-Aware Story Builder", page_icon="💬")
st.title("💌 Meena & Mathur: A Context-Aware Love Story Generator")

# Load OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load Pinecone vector store
logging.info("Initializing Pinecone vector store...")
docsearch = PineconeVectorStore.from_existing_index(
    index_name="hii",
    embedding=embedding
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Chat history initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User inputs
user_prompt = st.chat_input("Enter your story continuation instruction:")
generation_mode = st.radio("Choose generation mode:", ["Single Slide (250–300 words)", "Full Story (1500 words)"])

# Display chat messages
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# Process new input
if user_prompt:
    st.session_state.chat_history.append(("user", user_prompt))

    with st.chat_message("user"):
        st.markdown(user_prompt)

    retrieved_docs = retriever.invoke(user_prompt)

    if not retrieved_docs:
        response_text = "⚠️ No relevant slide found."
        st.session_state.chat_history.append(("assistant", response_text))
        with st.chat_message("assistant"):
            st.warning(response_text)
    else:
        most_relevant_slide_doc = retrieved_docs[0] if isinstance(retrieved_docs, list) else retrieved_docs
        relevant_slide_content = most_relevant_slide_doc.page_content

        base_prompt = (
            "You are continuing a love story between Meena and Mathur.\n"
            f"Previously, {relevant_slide_content.rstrip('.')}.\n"
        )

        if generation_mode == "Single Slide (100-150 words)":
            final_prompt = base_prompt + "Now, write the next slide with more emotional tension and vulnerability. Limit to 150 words."
            max_tokens = 500
        else:
            final_prompt = base_prompt + "Now, write a full story continuation (1500 words) with rich emotion and depth."
            max_tokens = 2000

        with st.chat_message("assistant"):
            st.markdown(f"📌 **Most Relevant Slide:**\n> {relevant_slide_content}")
            st.markdown(f"🧠 **Generating Story:** _{generation_mode}_")

            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=0.7,
                    max_tokens=max_tokens
                )
                generated_text = response.choices[0].message.content
                st.markdown(generated_text)
                st.session_state.chat_history.append(("assistant", generated_text))

            except Exception as e:
                error_msg = f"❌ Error generating text: {e}"
                st.error(error_msg)
                st.session_state.chat_history.append(("assistant", error_msg))
