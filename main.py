import os
from tempfile import NamedTemporaryFile
import streamlit as st
from langchain.agents import initialize_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from requests.exceptions import HTTPError

from tools import ImageCaptionTool, ObjectDetectionTool

# Initialize agent
tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

# Get API key from environment variable or directly specify it
openai_api_key = os.getenv('OPENAI_API_KEY', 'OPEN_KEY')

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0,
    model_name="gpt-3.5-turbo"
)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate'
)

# Streamlit UI
st.title('Ask a question to an image')
st.header("Please upload an image")

file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if file:
    st.image(file, use_column_width=True)
    user_question = st.text_input('Ask a question about your image:')

    if user_question:
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.getbuffer())
            image_path = temp_file.name

            with st.spinner(text="In progress..."):
                try:
                    response = agent.run(f'{user_question}, this is the image path: {image_path}')
                    st.write(response)
                except HTTPError as e:
                    if e.response.status_code == 429:
                        st.error("API quota exceeded. Please check your OpenAI plan and billing details.")
                    else:
                        st.error(f"HTTP error occurred: {e}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
