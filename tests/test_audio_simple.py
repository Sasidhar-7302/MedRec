
import asyncio
import edge_tts
import os

async def main():
    text = "This is a test of edge tts audio generation."
    voice = "en-US-GuyNeural"
    output = "test_audio.mp3"
    
    print(f"Generating {output} in CWD: {os.getcwd()}...")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output)
    
    if os.path.exists(output):
        print(f"SUCCESS: {output} exists. Size: {os.path.getsize(output)} bytes.")
    else:
        print(f"FAILURE: {output} does not exist.")

if __name__ == "__main__":
    asyncio.run(main())
