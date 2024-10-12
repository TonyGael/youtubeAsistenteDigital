
# Análisis del Código de Asistente de Voz

Este documento reúne un análisis detallado del código que implementa un asistente de voz utilizando el modelo GPT-2 y bibliotecas de reconocimiento de voz y texto a voz.

## Importaciones de Bibliotecas

```python
import speech_recognition as sr
from gtts import gTTS
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
```

1. **`import speech_recognition as sr`**: Importa la biblioteca `speech_recognition`, que permite la conversión de voz a texto mediante diversos servicios de reconocimiento de voz, incluido Google Speech Recognition. Se utiliza para capturar y procesar audio en tiempo real.

2. **`from gtts import gTTS`**: Importa `gTTS` (Google Text-to-Speech), que convierte texto en audio. Permite que el asistente responda de forma auditiva a las preguntas del usuario.

3. **`import os`**: Importa el módulo `os`, que proporciona funciones para interactuar con el sistema operativo. Se usa para manejar la reproducción de archivos de audio.

4. **`from transformers import GPT2LMHeadModel, GPT2Tokenizer`**: Importa clases específicas de la biblioteca `transformers`, que permiten cargar y utilizar el modelo GPT-2 y su tokenizador, respectivamente. Esto es esencial para generar texto basado en las preguntas del usuario.

5. **`import torch`**: Importa PyTorch, que es un marco de trabajo para aprendizaje profundo. Se utiliza aquí para manejar tensores y facilitar operaciones relacionadas con el modelo.

## Inicialización del Modelo y Tokenizador

```python
model_name = 'datificate/gpt2-small-spanish'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
```

1. **`model_name = 'datificate/gpt2-small-spanish'`**: Define el nombre del modelo que se utilizará, en este caso, una versión de GPT-2 entrenada en español. Esto es crucial para asegurar que el modelo entienda y genere texto en el idioma adecuado.

2. **`tokenizer = GPT2Tokenizer.from_pretrained(model_name)`**: Carga el tokenizador correspondiente al modelo especificado. El tokenizador convierte texto en tokens que el modelo puede procesar y viceversa.

3. **`model = GPT2LMHeadModel.from_pretrained(model_name)`**: Carga el modelo preentrenado GPT-2 utilizando el nombre especificado. Esto proporciona el modelo que generará respuestas basadas en las preguntas formuladas.

## Configuración de Tokens Especiales

```python
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
```

1. **`tokenizer.pad_token = tokenizer.eos_token`**: Establece el token de padding como el token de final de secuencia. Esto asegura que el modelo sepa cómo manejar secuencias de longitud variable durante la generación de texto.

2. **`model.config.pad_token_id = tokenizer.pad_token_id`**: Configura el ID del token de padding en la configuración del modelo. Esto garantiza que el modelo utilice el token correcto para el relleno de secuencias, evitando problemas durante la generación.

## Función de Escucha

```python
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
```

1. **Inicialización de Componentes**: Se crea un objeto `Recognizer` para manejar el reconocimiento de voz y un objeto `Microphone` para capturar audio desde el micrófono.

2. **Calibración y Escucha**: Se calibra el micrófono para el ruido ambiental y se escucha el audio del usuario. Esto permite mejorar la precisión del reconocimiento de voz.

3. **Reconocimiento de Voz**: Se intenta reconocer el audio capturado mediante el servicio de Google. Si tiene éxito, se devuelve el texto; de lo contrario, se manejan excepciones para proporcionar retroalimentación adecuada al usuario.

## Función de Hablar

```python
def speak(text):
    tts = gTTS(text=text, lang='es', slow=False)
    # Guardar el audio en un archivo temporal
    tts.save("temp.mp3")
    # Reproducir el audio
    os.system("mpg321 temp.mp3")  # Asegúrate de tener instalado mpg321 o usa un reproductor de audio de tu preferencia
```

1. **Conversión de Texto a Voz**: Se utiliza `gTTS` para convertir el texto proporcionado en audio en español.

2. **Guardado y Reproducción**: El audio generado se guarda como un archivo temporal y se reproduce utilizando un comando del sistema. Esto permite que el asistente responda auditivamente.

## Función de Preguntar

```python
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
```

1. **Codificación de la Pregunta**: La pregunta se convierte en IDs de tokens que el modelo puede procesar.

2. **Máscara de Atención**: Se crea una máscara de atención para indicar cuáles tokens son relevantes durante la generación de la respuesta.

3. **Generación de Respuesta**: Se llama al método `generate` del modelo para producir una respuesta basada en la pregunta. Se establecen varios parámetros para controlar la longitud y calidad de la respuesta.

4. **Decodificación**: La respuesta generada se decodifica de vuelta a texto legible para el usuario.

## Bloque Principal de Ejecución

```python
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
```

1. **Comprobación de Ejecución Principal**: Este bloque se ejecuta solo si el script se ejecuta directamente. Esto permite la reutilización del código en otros scripts sin ejecutar el bloque principal.

2. **Bucle Infinito**: Se establece un bucle que espera entradas del usuario de forma continua.

3. **Captura y Procesamiento**: Se captura el audio del usuario, se verifica si hay texto, y se permite al usuario salir diciendo "salir".

4. **Generación de Respuesta**: Si hay texto, se genera una respuesta utilizando la función `ask_question()` y se proporciona retroalimentación al usuario mediante la impresión y la reproducción de audio.

## Consideraciones Finales

Este asistente de voz integra múltiples tecnologías para permitir la interacción fluida entre el usuario y el modelo de lenguaje. La capacidad de escuchar, procesar preguntas y responder en voz alta hace que la experiencia sea dinámica y accesible.

---
