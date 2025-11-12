import re
import os
import sounddevice as sd
import queue
import wave
import numpy as np
import time
import json
import vosk
import threading
from llama_cpp import Llama
from piper import PiperVoice as tts
from Face_recognition_app import FaceRecognition
from RealtimeTTS import TextToAudioStream, PiperEngine, PiperVoice

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "piper", "en_GB-semaine-medium.onnx")
piper_voice = tts.load(model_path)
# --- Piper Setup ---
VOICE_MODEL = os.path.join(base_dir, "piper", "en_GB-semaine-medium.onnx")
VOICE_CONFIG = os.path.join(base_dir, "piper", "en_GB-semaine-medium.onnx.json")
PIPER_EXE = os.path.join(base_dir, "piper", "piper.exe")

voice = PiperVoice(model_file=VOICE_MODEL, config_file=VOICE_CONFIG)
engine = PiperEngine(piper_path=PIPER_EXE, voice=voice)
q=queue.Queue()

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model","gemma-3-4b-it-Q4_K_M.gguf")

llm = Llama(
    model_path=model_path,
    verbose=False,
    n_ctx=2048,
    chat_format='gemma'
)

model = vosk.Model("vosk-model-small-en-in-0.4")
samplerate = 16000
rec = vosk.KaldiRecognizer(model, samplerate)

#---------Function to play text with piper voice-------
def play(target_file):
    with wave.open(target_file,'rb') as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        n_frames = wav.getnframes()
        audio = wav.readframes(n_frames)
        audio_np = np.frombuffer(audio, dtype=np.int16)

        if n_channels == 2:
            audio_np = audio_np.reshape(-1, 2)

        audio_np = audio_np.astype(np.float32) / 32768.0
        sd.play(audio_np,sample_rate)
        sd.wait()

#This is the function that converts text to speech
def speak(text):
    file = 'Ai_test.wav'
    with wave.open(file, "wb") as wav_file:
        piper_voice.synthesize_wav(text, wav_file)
    play(file)

q = queue.Queue()
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))  # put audio data into queue

def listen(prompt=''):
    if prompt:  # Only speak and print if there's a prompt
        speak(prompt)
        print(prompt)
    while not q.empty():  # Clear any old audio
        q.get_nowait()
    last_activity=time.time()
    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback,dtype='int16'):
        while True:
            if time.time() - last_activity > 15:
                return False
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    print(text)
                    return text 
                
sys_prompt = """You are StudyBot — a kind, intelligent, and caring AI tutor designed to help students aged 5–15 understand science, math, and general knowledge.
Your personality:
- Cheerful, patient, and encouraging.
- Use short, clear sentences with examples.
- When a user asks about a topic give the simple overview first. Then ask if they need more detail.
- You must care about the student's learning experience above all else.
- If user wants to play quiz , ask the topic first , then ask question one by one and keep track of scores.
-At the end tell the score.
Your capabilities:
- You can see and understand both images and text.
- When an image is provided, first describe what you see in a single sentence,
  then connect it to the question or topic.
- When no image is provided, reason purely from the text.
Your response format:
1. Explain clearly (one or two short paragraphs).
2. Encourage the student with a follow-up question or idea.
3.In case of quiz , you ask topic  and then a questison.User replies with answer.Keep track and then ask the next question.
5.At the end ask user if they want to learn more on the topic.
Your goal:
To make learning fun, visual, and interactive and take care of your student like a friendly tutor would.
Note:
There could be mistakes in user input so ask if you don't understand.Use a maximum of 150 words.Don't repeat sentences"""

conversation = [{"role": "system", "content": sys_prompt}]


def chat(user_input):
    print("\nStudyBot:", end=" ", flush=True)
    conversation.append({"role": "user", "content": user_input})
    full_reply = ""  # <-- store complete response

    for chunk in llm.create_chat_completion(
        messages=conversation,
        stream=True,
    ):
        delta = chunk["choices"][0]["delta"]
        text = delta.get("content") or ""
        clean_text = re.sub(r'[*_#`~<>]', '', text)
        clean_text = clean_text.replace("**", "").replace("__", "")

        if clean_text:
            full_reply += clean_text
            yield clean_text  # streaming to TTS
            print(clean_text, end="", flush=True)

    # Append the **entire response**, not just last chunk
    conversation.append({"role": "assistant", "content": full_reply})
    print()  # newline after finishing


def main():
    wakeup='wake_up2.wav'
    sleep="sleep_sound.wav"
    mode='sleep'
    fr = FaceRecognition()
    speak("Hi! I am your study mate. My name is lili. Whenever you need me just call hello lili.")
    while True:
        if mode =='sleep':
            cmd=listen()
            print("Listening...")
            if  cmd and ("hi lily" in cmd or "hello lily" in cmd or "lily" in cmd):
                user = fr.run_recognition()
                play(wakeup)
                time.sleep(1)
                mode='awake'
                if user:
                    speak(f"Hello {' and '.join(user)}! How's it going. What do you want to know?")
                else:
                    speak("How can I assist you today?")
                continue
            
        elif mode=='awake':
            user_input = listen()  # Removed print of user_input
            if not user_input:
                speak("okay... byee.")
                time.sleep(1)
                play(sleep)
                mode = 'sleep'
                continue
            if "sleep" in user_input or "goodbye" in user_input or "good bye" in user_input:
                mode = 'sleep'
                speak("Okay,byeee")
                time.sleep(1)
                play(sleep)
                continue  
            else:
                # Generate response with optimized parameters
                try:
                    text_stream= chat(user_input)
                    TextToAudioStream(engine).feed(text_stream).play()
                except Exception as e:
                    print(f"Error generating response: {e}")
                    speak("I'm sorry, I couldn't process that properly.")

        

if __name__ == "__main__":
    main()
        

