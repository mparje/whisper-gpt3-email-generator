import streamlit as st
import os
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import torch
import numpy as np
import openai
import pyperclip

def record(duration):
    fs = 44100  # this is the frequency sampling; also: 4999, 64000
    seconds = duration  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    st.write("Grabando audio, por favor hable ahora...")
    sd.wait()  # Wait until recording is finished
    st.write("Grabación finalizada.")
    write('output.mp3', fs, myrecording)  # Save as MP3 file

def transscribe():
    torch.cuda.is_available()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = whisper.load_model("base", device=DEVICE)
    result_text = ""

    with st.spinner("Transcribiendo el audio..."):
        audio = whisper.load_audio('output.mp3')
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        _, probs = model.detect_language(mel)
        st.write(f"Idioma detectado: {max(probs, key=probs.get)}")

        options = whisper.DecodingOptions(language="en")
        result = whisper.decode(model, mel, options)
        result_text = result.text

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
