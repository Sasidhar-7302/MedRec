
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

print("Available Voices:")
for voice in voices:
    print(f"ID: {voice.id}")
    print(f"Name: {voice.name}")
    print(f"Gender: {voice.gender}") # Note: Gender attribute might not be available or reliable on all systems
    print("---")
