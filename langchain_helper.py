from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


def create_db_from_youtube_video_url(video_url: str, openai_api_key: str) -> FAISS:
    try:
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
        transcript = loader.load()
        if not transcript:
            raise ValueError("No transcript found. The video may not have captions.")
    except Exception as e:
        raise RuntimeError(f"Failed to load YouTube video: {str(e)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(docs, embeddings)
    return db, transcript[0].metadata


def get_response_from_query(db, query, openai_api_key, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key,
        temperature=0.7
    )

    prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that answers questions about YouTube videos
        based on the video's transcript.

        Answer the following question: {question}
        By searching the following video transcript: {docs}

        Only use the factual information from the transcript to answer the question.
        If you don't have enough information, say "I don't know".

        Your answers should be verbose and detailed.
    """)

    chain = prompt | llm

    response = chain.invoke({"question": query, "docs": docs_page_content})
    return response.content.replace("\n", ""), docs