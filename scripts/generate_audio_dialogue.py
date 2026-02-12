
import pyttsx3
import os
import wave

# Voices
VOICE_MALE = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0"
VOICE_FEMALE = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"

# Dialogue (Kaggle-style GI consultation)
dialogue = [
    ("Doctor", "Good morning, Ms. Doe. I understand you've been having some stomach issues?"),
    ("Patient", "Yes, doctor. I've had this burning pain in the middle of my stomach for months."),
    ("Doctor", "Does the pain move anywhere, like to your back?"),
    ("Patient", "Sometimes it goes to my back, especially after I eat spicy food."),
    ("Doctor", "Any nausea or vomiting?"),
    ("Patient", "I feel sick sometimes, but I haven't thrown up. I tried taking Tums but they don't help much."),
    ("Doctor", "Okay. I'm going to press on your stomach. Does this hurt?"),
    ("Patient", "Ouch! Yes, right there."),
    ("Doctor", "That's tender in the epigastric region. Given your symptoms, I suspect GERD or possibly an ulcer."),
    ("Doctor", "I'd like to start you on Omeprazole 40mg daily and schedule an EGD to take a look inside."),
    ("Patient", "Okay, doctor. Thank you.")
]

def generate_audio():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    
    # Generate segments
    segment_files = []
    print("Generating dialogue segments...")
    for i, (speaker, text) in enumerate(dialogue):
        voice_id = VOICE_MALE if speaker == "Doctor" else VOICE_FEMALE
        engine.setProperty('voice', voice_id)
        
        filename = f"temp_{i}.wav"
        engine.save_to_file(text, filename)
        engine.runAndWait()
        segment_files.append(filename)

    # Stitch using wave module
    print("Stitching audio segments...")
    output_file = "kaggle_dialogue.wav"
    with wave.open(output_file, 'wb') as outfile:
        for i, filename in enumerate(segment_files):
            with wave.open(filename, 'rb') as infile:
                if i == 0:
                    outfile.setparams(infile.getparams())
                outfile.writeframes(infile.readframes(infile.getnframes()))
            os.remove(filename)

    print(f"Exporting final audio to {output_file}...")
    print("Done.")

if __name__ == "__main__":
    generate_audio()
