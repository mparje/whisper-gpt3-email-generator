import streamlit as st
import os
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import torch
import numpy as np
import openai
import pyperclip

from audio_recorder_streamlit import audio_recorder

working_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(working_dir)

openai.api_key = os.getenv("OPENAI_API_KEY")


def transcribe(audio_file):
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript


st.text("Whisper Transcription and Summarization")


st.sidebar.title("Whisper Transcription and Summarization")

# Explanation of the app
st.sidebar.markdown("""
        This is an app that allows you to. 
        """)

# tab record audio and upload audio

audio_bytes = audio_recorder(pause_threshold=180)
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # save audio file to mp3
    with open(f"audio_{timestamp}.mp3", "wb") as f:
        f.write(audio_bytes)





    return result_text

def generate_mail(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Write a kind email for this: I can not come to work today because im really sick\n\nHi there,\n\nI'm sorry for the short notice, but I won't be able to come in to work today. I'm really sick and need to rest. I'll be back to work tomorrow. Hope you all have a wonderful day.\n\nThanks,\n\n[Your Name]\n\n\nWrite a kind email for this: {text}",
        temperature=0.7,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text

def main():
    st.title("Transcripción de audio y generación de correo electrónico")
    duration = st.slider("Duración de la grabación en segundos:", 1, 30, 8)
    record(duration)
    text = transscribe()
    st.write("Texto transcrito: ")
    st.write(text)
    email = generate_mail(text)
    st.write("Correo electrónico generado: ")
    st.write(email)
    st.write("¡El correo electrónico se ha copiado en el portapapeles!")
    pyperclip.copy(email)

if __name__ == "__main__":
    main()
