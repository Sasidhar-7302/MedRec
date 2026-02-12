
import pyttsx3
import os

text = """
This is Doctor Smith performing a consultation for patient Jane Doe. 
The patient is a 45-year-old female presenting with a three-month history of epigastric pain and bloating. 
She reports the pain is burning in nature, worse after eating spicy foods, and non-radiating. 
She denies dysphagia, odynophagia, or unintended weight loss. 
Review of systems is positive for occasional nausea but negative for vomiting, melena, or hematochezia. 
Physical exam reveals mild tenderness in the epigastrium with no rebound or guarding. 
My assessment is GERD, rule out peptic ulcer disease. 
Plan is to start Omeprazole 40 milligrams daily for 8 weeks. 
I will also schedule an upper endoscopy to visualize the esophagus and stomach. 
Follow up in 2 months or sooner if symptoms worsen.
"""

# Initialize TTS engine
engine = pyttsx3.init()

# Set properties (optional)
engine.setProperty('rate', 150)    # Speed percent (can go over 100)
engine.setProperty('volume', 0.9)  # Volume 0-1

# Save to file
output_file = "sample.wav"
print(f"Generating realistic GI sample audio to {output_file}...")
engine.save_to_file(text, output_file)
engine.runAndWait()
print("Done.")
