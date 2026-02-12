import asyncio
import edge_tts
import os

# Dialogue (Kaggle Case 2: Biliary Colic)
dialogue = [
    ("male", "I was wondering if you could tell me a little bit about what brought you in to the Emergency Department today?"),
    ("female", "Yeah, so nice to meet you. I've been having this pain right in my abdomen. It's kind of like in the upper right area."),
    ("male", "OK, and so uh, when, where is this painting located exactly?"),
    ("female", "So it's just in the upper right corner of my abdomen, right below where the lungs are, and it, yeah, it's just I have this severe pain that's going on."),
    ("male", "OK, and how long is it been going on for?"),
    ("female", "So it's been going on for the last few days and it got worse today."),
    ("male", "OK, and does the pain radiate anywhere?"),
    ("female", "Uh no, it stays right in the in the spot that I told you right in the right upper corner."),
    ("male", "OK, and when did the pain start? Or if you could tell me what were you doing right prior to the pain starting?"),
    ("female", "So I think it started after just three days ago after I had a meal like I I think it was after lunch around half an hour or an hour after lunch."),
    ("male", "How would you describe the character or the quality of the pain?"),
    ("female", "So it's like a sharp, I would describe it as like a sharp pain."),
    ("male", "OK, and on a scale of 1 to 10, 10 being the most severe pain, what would you rate it as?"),
    ("female", "I would rate it as, right now I would rate it as an 8."),
    ("male", "OK, and has there been anything that you've tried to make this pain any better?"),
    ("female", "I tried taking just like Advil and Tylenol, but it didn't really seem to help the pain too much."),
    ("male", "OK, and have you had any other associated symptoms such as nausea or or vomiting?"),
    ("female", "I've I've had some nausea over the past few days, but I haven't vomited anything."),
    ("male", "Do you take any medications on a regular basis?"),
    ("female", "Sometimes I take like some antacids when I get heartburn."),
    ("male", "OK, and any family history of gallbladder disease or cardiovascular disease in the family?"),
    ("female", "Um, so my father died of a stroke when he was in his 60s, my mother does have gallstones."),
    ("male", "OK, and do you drink alcohol?"),
    ("female", "Uh, yeah sometimes, maybe one or two glasses of wine every night.")
]

async def generate():
    segments = []
    for i, (gender, text) in enumerate(dialogue):
        voice = "en-US-GuyNeural" if gender == "male" else "en-US-JennyNeural"
        filename = f"seg_{i}.mp3"
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(filename)
        segments.append(filename)
        print(f"Generated segment {i+1}/{len(dialogue)}")
    
    # Combine by simple binary concatenation (works for MP3)
    try:
        with open("kaggle_dialogue.mp3", "wb") as outfile:
            for seg in segments:
                with open(seg, "rb") as infile:
                    outfile.write(infile.read())
        print("Created kaggle_dialogue.mp3 by binary concatenation")
        
        # Try to rename to wav if needed, but keeping as mp3 is safer for now
        # web_app can just load mp3
    except Exception as e:
        print(f"Concatenation failed: {e}")
    
    # Cleanup
    for seg in segments:
        try: os.remove(seg)
        except: pass
    try: os.remove("segments.txt")
    except: pass


if __name__ == "__main__":
    asyncio.run(generate())
