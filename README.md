# 🖼️ Vision2Voice AI — Image to Speech GenAI Tool Using LLM

Convert images into short AI-generated stories with **audio narration**.  
Vision2Voice AI uses **Hugging Face vision & TTS models**, **OpenAI GPT**, and **LangChain**, wrapped in a **Streamlit** app.

https://vision2voice-ai-k9kyxexpdkctthlakdld7r.streamlit.app/

Deployed on **Streamlit Cloud** and **Hugging Face Spaces** (self-host friendly).

---

## 🎥 Demo Previews

All sample **input images** and their **generated audio files** are stored in the `img-audio/` folder.

You can explore them here:
- Input + output pairs: `img-audio/`

### Example Outputs

> Screenshots below are just visual previews — the actual images and `.wav` files are inside `img-audio/`.

**Demo 1 — Couple Image**

<img width="300" alt="OUTPUT1" src="https://github.com/user-attachments/assets/542044ff-9f11-4e35-bb83-7f225c214204" />

**Demo 2 — Picnic Vacation Image**

<img width="300" alt="OUTPUT2" src="https://github.com/user-attachments/assets/f57953fb-bdf0-47ec-b168-adc8d68b942d" />

**Demo 3 — Family Image**

<img width="300" alt="OUTPUT3" src="https://github.com/user-attachments/assets/e2594a7b-e6c5-49e1-95b5-3f3f11787e3f" />

👉 For each demo, you can:
- View the **input image** in `img-audio/`
- Listen to the **generated audio story** (`.wav` or `.mp3`) with the same name.

---

## 🧱 System Design

The high-level architecture is shown below:

![System Design](img/system-design.drawio.png)

**Flow:**

1. **User uploads an image**
2. **Image → Caption** via Hugging Face **BLIP** model
3. **Caption → Short Story** via **OpenAI GPT** (through LangChain)
4. **Story → Speech** via Hugging Face **TTS** model
5. **Streamlit UI** displays:
   - Generated caption
   - AI short story
   - Playable audio narration

---

## 🧠 Approach

This app uses a 3-step AI pipeline:

1. ### 🖼️ Image → Text (Captioning)
   - Uses an image-to-text transformer model:  
     [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base)  
   - Generates a **scene description / caption** from the uploaded image.

2. ### ✍️ Text → Story (LLM)
   - Uses **OpenAI GPT (gpt-3.5-turbo or compatible)**  
   - Prompted via **LangChain** to turn the caption into a **short story**  
     (default ~**50 words**, configurable as needed).

3. ### 🔊 Story → Speech (TTS)
   - Uses a text-to-speech model:  
     [`espnet/kan-bayashi_ljspeech_vits`](https://huggingface.co/espnet/kan-bayashi_ljspeech_vits)  
   - Generates an **audio narration file** for the story.

4. ### 💻 User Interface (Streamlit)
   - Simple web UI where the user can:
     - Upload an image
     - See the generated caption and story
     - Play or download the audio file

---

## ✨ Features

- 🔁 **End-to-end pipeline**: image ➜ caption ➜ story ➜ speech
- 🤝 **Hybrid AI stack**: OpenAI GPT + Hugging Face models
- 🌐 **Streamlit UI**: easy to use in browser
- 🎧 **Audio outputs saved**: sample outputs in `img-audio/`
- 🧩 **Configurable story length & style** (prompt-tunable)
- 🧪 Included **demo images + audio** to quickly understand behavior

---

## 🧰 Tech Stack

- **Language:** Python 3
- **Framework:** Streamlit
- **LLM Orchestration:** LangChain
- **Models:**
  - Vision: BLIP (`Salesforce/blip-image-captioning-base`)
  - LLM: OpenAI GPT (`gpt-3.5-turbo` or similar)
  - TTS: VITS (`espnet/kan-bayashi_ljspeech_vits`)
- **Infra:** Local / Streamlit Cloud / Hugging Face Spaces

---

## 📦 Requirements

Main libraries (see `requirements.txt`):

- `os`
- `python-dotenv`
- `transformers`
- `torch`
- `langchain`
- `openai`
- `requests`
- `streamlit`

Install them all with:

```bash
pip install -r requirements.txt
