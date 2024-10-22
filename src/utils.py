import os
from key import groq_api_key

import wave
import pyaudio
from scipy.io import wavfile
import numpy as np

import whisper

from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from gtts import gTTS
import pygame

os.environ["GROQ_API_KEY"] = groq_api_key

def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    # Find the maximum absolute amplitude in the audio data
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold


def record_audio_chunk(audio, stream, chunk_length=5):
    print("Recording...")
    frames = []
    # Calculate the number of chunks needed for the specified length of recording
    # 16000 Hertz -> sufficient for capturing the human voice
    # 1024 frames -> the higher, the higher the latency
    num_chunks = int(16000 / 1024 * chunk_length)

    # Record the audio data in chunks
    for _ in range(num_chunks):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = './temp_audio_chunk.wav'
    print("Writing...")
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))  # Sample width
        wf.setframerate(16000)  # Sample rate
        wf.writeframes(b''.join(frames))  # Write audio frames

    # Check if the recorded chunk contains silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")


def load_whisper():
    model = whisper.load_model("base")
    return model


def transcribe_audio(model, file_path):
    print("Transcribing...")
    # Print all files in the current directory
    print("Current directory files:", os.listdir())
    if os.path.isfile(file_path):
        results = model.transcribe(file_path) # , fp16=False
        return results['text']
    else:
        return None

def load_prompt():
    input_prompt = """

    As an expert advisor specializing in make Q&A for quality month celebration 

    for example:
    For a "Choose the Best Answer" format in a chatbot for a Quality Month celebration, here’s a sample Q&A set with multiple-choice options:
Sample Q&A with Multiple Choice Options
1. Question:

Why is quality important in our organization?

Choose the best answer:

    A) It reduces the number of employees.
    B) It ensures our products and services meet customer expectations, increases efficiency, and builds trust.
    C) It is only important for compliance with regulations.

Correct answer: B
2. Question:

What does "Continuous Improvement" mean in a quality context?

Choose the best answer:

    A) Constantly redoing the same tasks without change.
    B) An ongoing effort to make incremental improvements in processes and products over time.
    C) Making large changes all at once every few years.

Correct answer: B
3. Question:

What are the key benefits of maintaining high-quality standards?

Choose the best answer:

    A) Increased customer satisfaction, lower costs, and better compliance.
    B) Only compliance with government regulations.
    C) Higher costs and more work for employees.

Correct answer: A
4. Question:

What is a "root cause analysis" and why is it important in quality management?

Choose the best answer:

    A) It helps to identify the main cause of a problem, preventing it from recurring.
    B) It is used to avoid solving the problem and focuses only on symptoms.
    C) It blames individuals for problems in processes.

Correct answer: A
5. Question:

Which of the following is NOT one of the 7 Quality Tools?

Choose the best answer:

    A) Fishbone Diagram
    B) Control Chart
    C) Financial Report

Correct answer: C
6. Question:

What is ISO 9001?

Choose the best answer:

    A) An international standard that defines environmental management systems.
    B) A certification focused on health and safety at work.
    C) A standard for quality management systems ensuring consistent product and service quality.

Correct answer: C
7. Question:

How can employees best contribute to quality improvement?

Choose the best answer:

    A) By following processes, suggesting improvements, and being mindful of quality.
    B) By keeping quality issues to themselves.
    C) By reporting problems only when asked.

Correct answer: A
8. Question:

What does the PDCA cycle stand for in quality management?

Choose the best answer:

    A) Plan-Do-Check-Act
    B) Predict-Deliver-Control-Audit
    C) Process-Develop-Communicate-Analyze

Correct answer: A
9. Question:

What is the main difference between Quality Assurance (QA) and Quality Control (QC)?

Choose the best answer:

    A) QA focuses on defect prevention, while QC focuses on defect detection.
    B) QA and QC are the same processes.
    C) QC is done before production, and QA is done after.

Correct answer: A
10. Question:

Why is customer feedback important for quality improvement?

Choose the best answer:

    A) It helps identify areas where products or services may not meet customer expectations, allowing for targeted improvements.
    B) It helps gather complaints for record-keeping only.
    C) It is useful only for marketing purposes.

Correct answer: A

This format will keep participants engaged while also promoting learning about quality concepts. You can adjust the questions and options as needed for your specific organization or audience.
    
     and don't chat with yourself!. validate answer after 5 quest give marks  to human

    Previous conversation:
    {chat_history}

    New human question: {question}
    Response:
    """
    return input_prompt


def load_llm():
    chat_groq = ChatGroq(temperature=0, model_name="llama3-8b-8192",
                         groq_api_key=groq_api_key)
    return chat_groq


def get_response_llm(user_question, memory):
    input_prompt = load_prompt()

    chat_groq = load_llm()

    # Look how "chat_history" is an input variable to the prompt template
    prompt = PromptTemplate.from_template(input_prompt)

    chain = LLMChain(
        llm=chat_groq,
        prompt=prompt,
        verbose=True,
        memory=memory
    )

    response = chain.invoke({"question": user_question})

    return response['text']


def play_text_to_speech(text, language='en', slow=False):
    # Generate text-to-speech audio from the provided text
    tts = gTTS(text=text, lang=language, slow=slow)

    # Save the generated audio to a temporary file
    temp_audio_file = "temp_audio.mp3"
    tts.save(temp_audio_file)

    # Initialize the pygame mixer for audio playback
    pygame.mixer.init()

    # Load the temporary audio file into the mixer
    pygame.mixer.music.load(temp_audio_file)

    # Start playing the audio
    pygame.mixer.music.play()

    # Wait until the audio playback finishes
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Control the playback speed

    # Stop the audio playback
    pygame.mixer.music.stop()

    # Clean up: Quit the pygame mixer and remove the temporary audio file
    pygame.mixer.quit()
    os.remove(temp_audio_file)