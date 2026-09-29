import streamlit as st 
import validators
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.document_loaders import UnstructuredURLLoader

st.set_page_config(
    page_title="Youtube & Website Summarizer",
    page_icon="📝"
)

st.title("📝 Youtube & Website Summarizer")

st.write(
    "Summarize YouTube videos and Website pages "
    "using LangChain + Hugging Face."
)

with st.sidebar:
    hf_api_key = st.text_input("Huggingface API Token", type="password")
    
generic_url = st.text_input("Enter Youtube or Website URL")

prompt = PromptTemplate.from_template(
        """
        You are an expert summarization assistant.

        Summarize the following content clearly.

        Include:
        - Main topic
        - Important points
        - Key facts
        - Conclusion

        Content:
        {text}

        Summary:
        """    
)

if st.button("Summarize"):
    if not hf_api_key.strip():
        st.error("Please provide your GROQ API Key.")
        st.stop()

    if not generic_url.strip():
        st.error("Please provide a URL.")
        st.stop()

    if not validators.url(generic_url):
        st.error("Please enter a valid URL.")
        st.stop()
        
    llm = HuggingFaceEndpoint(
        repo_id= "meta-llama/Llama-3.1-8B-Instruct",
        huggingfacehub_api_token=hf_api_key,
        temperature=0.3,
        max_new_tokens=512
    )
    
    chat_model= ChatHuggingFace(llm=llm)
    
    chain = (
        prompt
        | chat_model
        | StrOutputParser()
    )
    
    try:
        ## Youtube
        if ("youtube.com" in generic_url or "youtu.be" in generic_url):
            st.info("Youtube video detected.")
            # Extract video id
            if "v=" in generic_url:
                video_id = (
                    generic_url.split("v=")[1].split("&")[0]
                )
            else:
                video_id = (
                    generic_url.split("youtu.be/")[1].split("?")[0]
                )
            # Geting the transcript
            api = YouTubeTranscriptApi()
            transcript = api.fetch(video_id=video_id)
            text = " ".join(
                item.text for item in transcript
            )
        ## Website
        else:
            st.info("Website Detected")
            loader = UnstructuredURLLoader(urls=[generic_url])
            docs = loader.load()
            text = "\n\n".join(
                doc.page_content for doc in docs
            )
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 2000, chunk_overlap=200
        )
        chunks = splitter.split_text(text)
        
        ## Summarize
        summaries = []
        with st.spinner("Genrating Summary..."):
            for chunk in chunks[:3]:
                result = chain.invoke(
                    {
                        "text":chunk
                    }
                )
                summaries.append(result)
            # Combine chunk summaries
            combined = "\n\n".join(summaries)
            final_summary = chain.invoke(
                {
                    "text": combined
                }
            )
        st.subheader("📄 Summary")
        st.write(final_summary)
        
    except Exception as e:
        st.error(f"Error: {e}")