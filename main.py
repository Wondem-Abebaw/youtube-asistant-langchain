import streamlit as st
import langchain_helper as lch
import textwrap

st.title("🎥 YouTube Assistant")

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

if query and youtube_url:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Validate URL
    if "youtube.com/watch" not in youtube_url and "youtu.be/" not in youtube_url:
        st.error("Please enter a valid YouTube URL.")
        st.stop()

    try:
        with st.spinner("Loading transcript and building database..."):
            db, metadata = lch.create_db_from_youtube_video_url(youtube_url, openai_api_key)

        # Show video metadata
        if metadata.get("title"):
            st.markdown(f"**📺 Title:** {metadata.get('title')}")
        if metadata.get("author"):
            st.markdown(f"**👤 Author:** {metadata.get('author')}")

        with st.spinner("Generating answer..."):
            response, docs = lch.get_response_from_query(db, query, openai_api_key)

        st.subheader("💬 Answer:")
        st.text(textwrap.fill(response, width=85))

        # Show source chunks
        with st.expander("📄 View source transcript chunks used"):
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.divider()

    except RuntimeError as e:
        st.error(f"Error: {str(e)}")