import os
import time
from typing import Any

import requests
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

from utils.custom import css_code

# ================= ENV =================
load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ================= UI HELPERS =================
def progress_bar(amount_of_time: int) -> None:
    progress_text = "Please wait, Generative models hard at work..."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(amount_of_time):
        time.sleep(0.03)
        my_bar.progress(percent_complete + 1, text=progress_text)

    time.sleep(0.5)
    my_bar.empty()

# ================= BLIP MODEL =================
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model


def generate_text_from_image(uploaded_file) -> str:
    """
    Uses BLIP to generate a caption from an uploaded image
    """
    processor, model = load_blip()

    image = Image.open(uploaded_file).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption

# ================= STORY GENERATION =================
def generate_story_from_text(scenario: str) -> str:
    """
    Uses GPT-3.5 + LangChain to generate a short story
    """
    prompt_template = """
    You are a talented storyteller.
    Create a short story (max 50 words) based on the image description below.

    CONTEXT:
    {scenario}

    STORY:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["scenario"]
    )

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.9,
        openai_api_key=OPENAI_API_KEY
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.predict(scenario=scenario)

# ================= TEXT TO SPEECH =================
def generate_speech_from_text(message: str) -> None:
    """
    Uses ESPnet VITS TTS from Hugging Face Inference API
    """
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
    }
    payload = {"inputs": message}

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        with open("generated_audio.flac", "wb") as file:
            file.write(response.content)
    else:
        st.error("Text-to-Speech generation failed.")

# ================= MAIN APP =================
def main() -> None:
    st.set_page_config(
        page_title="IMAGE TO STORY CONVERTER",
        page_icon="🖼️"
    )

    st.markdown(css_code, unsafe_allow_html=True)

    with st.sidebar:
        st.write("---")
        st.write("AI App created by @Shivam Roy")

    st.header("🖼️ Image-to-Story Converter")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=600)

        progress_bar(80)

        scenario = generate_text_from_image(uploaded_file)
        story = generate_story_from_text(scenario)
        generate_speech_from_text(story)

        with st.expander("🧠 Image Caption"):
            st.write(scenario)

        with st.expander("📖 Generated Story"):
            st.write(story)

        st.audio("generated_audio.flac")

# ================= RUN =================
if __name__ == "__main__":
    main()
