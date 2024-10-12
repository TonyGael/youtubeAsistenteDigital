import speech_recognition as sr
from gtts import gTTS
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Inicializar el modelo GPT-2 y el tokenizador
model_name = 'datificate/gpt2-small-spanish'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Establecer pad_token_id como el token de final de secuencia
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id  # Establecer pad_token_id en el modelo

def listen():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Calibrando el micrófono...")
        recognizer.adjust_for_ambient_noise(source)
        print("Di algo:")
        audio = recognizer.listen(source)

    try:
        print("Reconociendo...")
        text = recognizer.recognize_google(audio, language="es-ES")
        print(f"Tu dijiste: {text}")
        return text
    except sr.UnknownValueError:
        print("No pude entender lo que dijiste")
        return None
    except sr.RequestError:
        print("No pude conectarme al servicio de reconocimiento de voz")
        return None

def speak(text):
    tts = gTTS(text=text, lang='es', slow=False)
    # Guardar el audio en un archivo temporal
    tts.save("temp.mp3")
    # Reproducir el audio
    os.system("mpg321 temp.mp3")  # Asegúrate de tener instalado mpg321 o usa un reproductor de audio de tu preferencia

def ask_question(question):
    input_ids = tokenizer.encode(question, return_tensors='pt')
    
    # Crear la attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    output = model.generate(
        input_ids,
        max_length=200,  # Establecer un valor alto para max_length, ajusta según sea necesario
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        attention_mask=attention_mask,  # Pasar la attention mask
    )
    
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    while True:
        text = listen()
        if text:
            if "salir" in text.lower():
                speak("Adiós")
                break

            response = ask_question(text)
            print(f"Respuesta: {response}")
            speak(response)
