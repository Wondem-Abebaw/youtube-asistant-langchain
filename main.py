import streamlit as st
import langchain_helper as lch
import textwrap

st.title("🎥 YouTube Assistant")

# Initialize session state for chat history and DB
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "db" not in st.session_state:
    st.session_state.db = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "last_url" not in st.session_state:
    st.session_state.last_url = None

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.text_area(
            label="YouTube Video URL",
            max_chars=100
        )
        query = st.text_area(
            label="Ask a question about the video",
            max_chars=200,
            key="query"
        )
        openai_api_key = st.text_input(
            label="OpenAI API Key",
            key="langchain_search_api_key_openai",
            type="password"
        )
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        submit_button = st.form_submit_button(label='Submit')

    # Clear chat history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.db = None
        st.session_state.metadata = None
        st.rerun()

if query and youtube_url and submit_button:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Validate URL
    if "youtube.com/watch" not in youtube_url and "youtu.be/" not in youtube_url:
        st.error("Please enter a valid YouTube URL.")
        st.stop()

    try:
        # Only build DB if URL changed or DB doesn't exist yet
        if st.session_state.db is None or st.session_state.get("last_url") != youtube_url:
            with st.spinner("Loading transcript and building database..."):
                st.session_state.db, st.session_state.metadata = lch.create_db_from_youtube_video_url(youtube_url, openai_api_key)
                st.session_state.last_url = youtube_url
                st.session_state.chat_history = []  # reset history for new video

        with st.spinner("Generating answer..."):
            response, docs = lch.get_response_from_query(
                st.session_state.db, query, openai_api_key
            )

        # Save to chat history
        st.session_state.chat_history.append({
            "question": query,
            "answer": response,
            "docs": docs
        })

    except RuntimeError as e:
        st.error(f"Error: {str(e)}")

# Show video metadata
if st.session_state.metadata:
    if st.session_state.metadata.get("title"):
        st.markdown(f"**📺 Title:** {st.session_state.metadata.get('title')}")
    if st.session_state.metadata.get("author"):
        st.markdown(f"**👤 Author:** {st.session_state.metadata.get('author')}")
    st.divider()

# Display full chat history
if st.session_state.chat_history:
    for i, chat in enumerate(st.session_state.chat_history):
        # User message
        with st.chat_message("user"):
            st.write(chat["question"])

        # Assistant message
        with st.chat_message("assistant"):
            st.text(textwrap.fill(chat["answer"], width=85))
            with st.expander("📄 View source transcript chunks used"):
                for j, doc in enumerate(chat["docs"]):
                    st.markdown(f"**Chunk {j+1}:**")
                    st.write(doc.page_content)
                    st.divider()
else:
    st.info("👆 Enter a YouTube URL and ask a question to get started.")