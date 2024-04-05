import os
import streamlit as st
from tempfile import NamedTemporaryFile
import cv2
import base64
import time
import io
import openai
import requests
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from PyPDF2 import PdfReader
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "Enter Yourapikey" 

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Set page configuration
st.set_page_config(page_title="GPT-4 Tools", page_icon=":robot_face:", initial_sidebar_state="expanded")

def pdf_query(pdf_file, query):
    """
    Processes a PDF file, extracts text, generates embeddings,
    performs document search, and answers questions using Langchain.

    Args:
        pdf_file (streamlit.UploadedFile): The uploaded PDF file.
        query (str): User query for the PDF content.

    Returns:
        str: Answer to the user query based on the PDF content.
    """

    # Save  uploaded PDF to a temp  file
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(pdf_file.read())
        pdf_path = tmpfile.name

    # Read text from the PDF
    pdfreader = PdfReader(pdf_path)
    raw_text = ""
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    # Text splitting using CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800, #Split Text into chunks
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    openai_api_key = os.getenv('OPENAI_API_KEY')
    embeddings = OpenAIEmbeddings()
   
    document_search = FAISS.from_texts(texts, embeddings)  # Document search using FAISS vector store-Finds Relevant passages

    # Load question answering chain with OpenAI
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

   
    docs = document_search.similarity_search(query)  # Search documents for relevant passages=query

    # Answer the user's question based on retrieved documents
    answer = chain.run(input_documents=docs, question=query)

    # Delete the temporary PDF file
    os.remove(pdf_path)

    return answer

def video_to_frames(video_file, max_duration=10):
    
    # Save the uploaded video file to a temporary file
    with NamedTemporaryFile(delete=False, suffix=".mp4") as tmpfile:
        tmpfile.write(video_file.read())
        video_filename = tmpfile.name

    # Load the video clip
    video_clip = VideoFileClip(video_filename)

    # Trim the video to the specified max duration
    if video_clip.duration > max_duration:
        video_clip = video_clip.subclip(0, max_duration)

    video_duration = video_clip.duration # trimmed video duration

    # Extract frames
    base64Frames = []
    for frame in video_clip.iter_frames():
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video_clip.close()

    print(len(base64Frames), "frames read.")
    return base64Frames, video_duration, video_filename

def frames_to_story(base64Frames, prompt, model="gpt-4-vision-preview"):
    
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                prompt,
                *map(lambda x: {"image": x, "resize": 428}, base64Frames[0::50]),
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 300,
    }

    response = openai.chat.completions.create(**params)
    print(response.choices[0].message.content)  # print and return the response

    return response.choices[0].message.content

def text_to_audio(text):
    response = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        },
        json={
            "model": "tts-1",
            "input": text,
            "voice": "fable",
        },
    )

    # Check if the request was successful
    if response.status_code != 200:
        raise Exception("Request failed with status code")

    # Create an in-memory bytes buffer for audio data
    audio_bytes_io = io.BytesIO()

    # Write audio data to the buffer
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        audio_bytes_io.write(chunk)

    # Important: Seek to the start of the BytesIO buffer before returning
    audio_bytes_io.seek(0)

    # Save audio to a temporary file
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            tmpfile.write(chunk)
        audio_filename = tmpfile.name

    return audio_filename, audio_bytes_io

def merge_audio_video(video_filename, audio_filename, output_filename):
    print("Merging audio and video...")
    print("Video filename:", video_filename)
    print("Audio filename:", audio_filename)

    # Load the video and audio clips
    video_clip = VideoFileClip(video_filename)
    audio_clip = AudioFileClip(audio_filename)

    # Set the audio of the video clip as the audio file
    final_clip = video_clip.set_audio(audio_clip)

    # Write the result to a file (without audio)
    final_clip.write_videofile(
        output_filename, codec="libx264", audio_codec="aac")

    # Close the clips
    video_clip.close()
    audio_clip.close()

    # Return the path to the new video file
    return output_filename

def main():
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
    except ValueError as e:
        st.error(e)
        return

    feature_choice = st.sidebar.selectbox("Choose a feature:", ("Video Vision-to-TTS", "PDF Query"))

    if feature_choice == "Video Vision-to-TTS":
        st.image("gp4 vision to tts img.jpeg",width=200)
        uploaded_file = st.file_uploader("Choose a video")
        if uploaded_file is not None:
            
            st.video(uploaded_file)
            prompt = st.text_area(
                "Prompt (describe the video)", value="These are frames of a quick product demo walkthrough. Create a short voiceover script that outlines the key actions to take, that can be used along this product demo.")

            if st.button('Generate', type="primary") and uploaded_file is not None:
                with st.spinner('Processing...'):
                    # Extract frames and get video duration
                    base64Frames, video_duration, video_filename = video_to_frames(
                        uploaded_file)

                    est_word_count = video_duration * 2
                    final_prompt = prompt + f"(This video is ONLY {video_duration} seconds long, so make sure the voice over MUST be able to be explained in less than {est_word_count} words)"

                    # Generate story prompt using frames and prompt
                    text = frames_to_story(base64Frames, final_prompt)
                    st.write(text)

                    # Generate audio from text
                    audio_filename, audio_bytes_io = text_to_audio(text)

                    # Merge audio and video
                    output_video_filename = os.path.splitext(video_filename)[
                        0] + '_output.mp4'
                    final_video_filename = merge_audio_video(
                        video_filename, audio_filename, output_video_filename)

                    # Display the result
                    st.video(final_video_filename)

                    # Clean up temporary files
                    os.unlink(video_filename)
                    os.unlink(audio_filename)
                    os.unlink(final_video_filename)

    elif feature_choice == "PDF Query":
        st.image("b image.jpeg", width=200) 
        pdf_file = st.file_uploader("Upload a PDF")
        if pdf_file is not None:
            query = st.text_area("Enter your query")
            if st.button("Search", type="primary"):
                with st.spinner("Searching..."):
                    try:
                        answer = pdf_query(pdf_file, query)
                        st.write("Answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
