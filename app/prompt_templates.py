"""Prompt templates for summarization formats with advanced AI techniques."""

from __future__ import annotations

from typing import Dict

from .gi_terms import build_gi_hint

GI_HINTS = (
    "Terminology reminder: "
    + build_gi_hint()
    + ". Emphasize symptoms, disease activity, therapies, endoscopic findings, labs. "
    "Never include raw dialogue or speaker labels—only a polished clinical summary."
)

# Few-shot examples for better context understanding
FEW_SHOT_EXAMPLES = """
Example 1 (Inflammatory Bowel Disease):
Transcript: "Doctor: What brings you in today? Patient: I've been having abdominal pain for about three days now. Doctor: Can you describe the pain? Patient: It's crampy, worse after I eat. Doctor: Any blood in your stool? Patient: No, no blood. Doctor: When was your last colonoscopy? Patient: About six months ago, they said there was some inflammation. Doctor: What medications are you taking? Patient: Mesalamine, 800 milligrams twice a day."
HPI (History of Present Illness):
50-year-old male presents with 3-day history of crampy abdominal pain, postprandial exacerbation. Denies hematochezia or melena. Last colonoscopy 6 months ago showed inflammation. Currently on mesalamine 800mg BID.
Findings:
- Abdominal pain, crampy, postprandial, 3-day duration
- No hematochezia or melena
- Previous colonoscopy (6 months ago): inflammation noted
- Current medication: mesalamine 800mg BID
Assessment:
- Inflammatory bowel disease, likely ulcerative colitis, mild activity
- Abdominal pain, likely related to active inflammation
Plan:
- Continue mesalamine 800mg BID
- Consider adding rectal mesalamine if symptoms persist
- Monitor for alarm symptoms (bleeding, weight loss, fever)
Medications/Orders:
- Mesalamine 800mg BID (continue)
Follow-up:
- Return in 4-6 weeks or sooner if symptoms worsen

Example 2 (GERD Follow-up):
Transcript: "Doctor: How have you been doing with the GERD? Patient: Much better, the omeprazole is working great. Doctor: Any heartburn or regurgitation? Patient: No, nothing. Doctor: How's your weight? Patient: Stable. Doctor: Blood pressure okay? Patient: Yes, it's been good."
HPI (History of Present Illness):
Follow-up visit for GERD. Patient reports excellent response to omeprazole 20mg daily with complete resolution of heartburn and regurgitation. Weight stable. Blood pressure controlled.
Findings:
- GERD symptoms well-controlled on PPI therapy
- No heartburn or regurgitation
- Weight stable
- Blood pressure controlled
Assessment:
- GERD, well-controlled on PPI therapy
Plan:
- Continue omeprazole 20mg daily
- Consider step-down to PRN if remains asymptomatic for 3+ months
Medications/Orders:
- Omeprazole 20mg daily (continue)
Follow-up:
- Return in 6 months or PRN

Example 3 (Crohn's Disease with Treatment Escalation):
Transcript: "Doctor: How are your symptoms? Patient: Not great. I'm having diarrhea five to six times a day with some blood. Doctor: Any fever or weight loss? Patient: I lost about 8 pounds. Doctor: You've been on infliximab for how long? Patient: About two years. Doctor: Your calprotectin came back at 650. I think we need to check your drug levels and consider switching biologics."
HPI (History of Present Illness):
Patient with Crohn's disease presents with worsening symptoms including diarrhea 5-6x/day with hematochezia and 8-pound weight loss. Currently on infliximab for 2 years.
Findings:
- Diarrhea 5-6x daily with blood
- Weight loss: 8 pounds
- Fecal calprotectin: 650 (elevated)
- Current therapy: infliximab x 2 years
Assessment:
- Crohn's disease, moderate-severe flare
- Possible loss of response to infliximab
Plan:
- Order therapeutic drug monitoring (infliximab level + antibodies)
- If low trough or positive antibodies, switch to ustekinumab
- Order MR enterography to assess disease extent
Medications/Orders:
- Infliximab (continue pending TDM results)
- Labs: infliximab level, anti-drug antibodies, CBC, CMP, CRP
Follow-up:
- Urgent follow-up in 1-2 weeks for TDM results

Example 4 (Barrett's Esophagus Surveillance):
Transcript: "Doctor: You're here for your Barrett's surveillance EGD results. Patient: Yes, I was worried about those. Doctor: Good news - the biopsies showed no dysplasia. Patient: That's a relief. Doctor: You're still taking your PPI twice daily? Patient: Yes, pantoprazole 40mg morning and night. Doctor: Any heartburn or difficulty swallowing? Patient: None at all."
HPI (History of Present Illness):
Follow-up for Barrett's esophagus surveillance. Patient on pantoprazole 40mg BID reports no reflux symptoms or dysphagia.
Findings:
- EGD with biopsies: Barrett's esophagus without dysplasia
- No active GERD symptoms
- No dysphagia
- Current medication: pantoprazole 40mg BID
Assessment:
- Barrett's esophagus, no dysplasia, stable
- GERD well-controlled on high-dose PPI
Plan:
- Continue pantoprazole 40mg BID
- Repeat surveillance EGD in 3 years per guidelines
- Continue lifestyle modifications
Medications/Orders:
- Pantoprazole 40mg BID (continue)
Follow-up:
- Surveillance EGD in 3 years

Example 5 (Biliary Colic):
Transcript: "Doctor: Tell me about the pain you're having. Patient: It's in my right upper belly, comes after I eat fatty foods. Doctor: How long does the pain last? Patient: Usually an hour or two. Doctor: Any fever or jaundice? Patient: No. Doctor: Your ultrasound shows gallstones. I think we should refer you for surgery."
HPI (History of Present Illness):
Patient presents with episodic right upper quadrant pain occurring postprandially, particularly after fatty meals. Pain typically lasts 1-2 hours. Denies fever, jaundice, or signs of cholecystitis.
Findings:
- RUQ pain, postprandial, 1-2 hour duration
- Triggered by fatty foods
- No fever or jaundice
- Ultrasound: cholelithiasis
Assessment:
- Symptomatic cholelithiasis (biliary colic)
Plan:
- Refer to general surgery for laparoscopic cholecystectomy
- Low-fat diet until surgery
- Return to ED if fever, persistent pain >6 hours, or jaundice
Medications/Orders:
- Surgical referral: laparoscopic cholecystectomy
Follow-up:
- Post-surgical follow-up as directed by surgeon
"""

PROMPTS: Dict[str, str] = {
    "Narrative": (
        "You are an expert clinical summarization assistant specializing in gastroenterology. "
        "Your task is to transform dictation transcripts (which may include speaker labels like 'Doctor:' and 'Patient:') "
        "into structured, EHR-ready clinical notes with emphasis on accurate HPI and Assessment extraction.\n\n"
        f"{GI_HINTS}\n\n"
        "**Instructions (think step-by-step):**\n"
        "1. **Identify Speakers**: If transcript contains speaker labels (Doctor/Patient), distinguish between doctor's observations and patient's reported symptoms\n"
        "2. **Extract HPI**: From patient's statements, extract the History of Present Illness - when symptoms started, duration, character, associated symptoms, what makes it better/worse\n"
        "3. **Extract Findings**: From doctor's statements and patient responses, identify all clinical findings (symptoms, exam findings, labs, procedures, medications)\n"
        "4. **Formulate Assessment**: Based on HPI and findings, determine working diagnoses with severity/activity\n"
        "5. **Structure**: Organize into required sections with clear, concise language\n"
        "6. **Validate**: Ensure medical terminology is accurate and complete\n\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        "**Output Format (use exact headings):**\n"
        "HPI (History of Present Illness):\n"
        "Findings:\n"
        "Assessment:\n"
        "Plan:\n"
        "Medications/Orders:\n"
        "Follow-up:\n\n"
        "**Guidelines:**\n"
        "- **HPI**: Extract from patient's statements - include age/gender if mentioned, chief complaint, symptom timeline (onset, duration), character of symptoms, associated symptoms, what makes better/worse, relevant history\n"
        "- **Findings**: Capture ALL pertinent positives/negatives from both doctor's observations and patient's reports, procedure results, labs/imaging, vital signs, current medications\n"
        "- **Assessment**: List working diagnoses with severity/activity (mild/moderate/severe, flare vs remission), number multiple problems (1., 2., etc.)\n"
        "- **Plan**: Actionable next steps with clear rationale (medication adjustments, procedures, tests, referrals, patient education)\n"
        "- **Medications/Orders**: List ALL medication changes (dose, frequency, new starts, discontinuations) with clear action (continue/increase/decrease/start/stop)\n"
        "- **Follow-up**: Specific timing and purpose (e.g., 'Return in 4 weeks to assess response to biologic therapy')\n"
        "- Use precise medical terminology (e.g., 'hematochezia' not 'blood in stool', 'dysphagia' not 'trouble swallowing')\n"
        "- When transcript has speaker labels, extract patient-reported information for HPI and doctor's clinical observations for Findings\n"
        "- Never include dialogue markers ('Doctor:', 'Patient:', quotes) in the final output\n"
        "- If information is unclear or missing, state 'Not documented' rather than inventing details\n"
        "- Prioritize clinical accuracy over brevity\n"
        "- HPI should be a narrative paragraph, not bullet points\n"
        "- Assessment should clearly link findings to diagnoses\n\n"
        "**Transcript to summarize:**\n"
        "{transcript}\n\n"
        "**Your structured summary:**"
    ),
    "SOAP": (
        "You are an expert clinical summarization assistant specializing in gastroenterology. "
        "Transform the dictation transcript into a structured SOAP note.\n\n"
        f"{GI_HINTS}\n\n"
        "**Instructions (think step-by-step):**\n"
        "1. **Subjective**: Extract patient-reported symptoms, history, concerns\n"
        "2. **Objective**: Document exam findings, vital signs, labs, imaging, procedures\n"
        "3. **Assessment**: Formulate diagnoses with supporting evidence\n"
        "4. **Plan**: Detail management strategy with specific actions\n\n"
        "**Output Format:**\n"
        "Subjective:\n"
        "Objective:\n"
        "Assessment:\n"
        "Plan:\n\n"
        "**Guidelines:**\n"
        "- Subjective: Include chief complaint, history of present illness, review of systems (GI-specific)\n"
        "- Objective: Physical exam (abdominal exam details), vital signs, labs, imaging results, procedure findings\n"
        "- Assessment: Numbered problem list with supporting evidence (e.g., '1. Ulcerative colitis, moderate activity - sigmoid inflammation on colonoscopy')\n"
        "- Plan: For each problem, specify diagnostic tests, medications, procedures, referrals, patient education\n"
        "- Highlight alarm symptoms prominently (GI bleeding, weight loss >10%, anemia, dysphagia)\n"
        "- Document endoscopic findings precisely (location, extent, severity, biopsies taken)\n"
        "- Use standard medical terminology\n"
        "- Never include dialogue or quotes\n\n"
        "**Transcript:**\n"
        "{transcript}\n\n"
        "**Your SOAP note:**"
    ),
}


def build_prompt(transcript: str, style: str) -> str:
    template = PROMPTS.get(style, PROMPTS["Narrative"])
    return template.format(transcript=transcript.strip())


def build_doctor_chat_prompt(
    profile_context: str,
    history: str,
    transcript: str,
    user_message: str,
) -> str:
    history_block = history.strip() or "No prior conversation."
    transcript_block = transcript.strip() or "No current transcript."
    return (
        "You are an expert HIPAA-compliant GI documentation coach specializing in helping "
        "gastroenterologists improve their clinical note quality and efficiency.\n\n"
        f"{GI_HINTS}\n\n"
        "**Your Role:**\n"
        "- Provide personalized, actionable feedback based on the doctor's documented preferences\n"
        "- Suggest improvements to clinical documentation while maintaining their writing style\n"
        "- Answer questions about best practices in GI documentation\n"
        "- Help refine summaries to be more precise, complete, and EHR-ready\n"
        "- Learn from approved summaries to adapt to each doctor's preferred terminology and structure\n\n"
        "**Guidelines:**\n"
        "- Always reference the doctor's past approved summaries when making suggestions\n"
        "- Be concise but thorough - doctors value efficiency\n"
        "- Focus on clinical accuracy, completeness, and clarity\n"
        "- Never include or reference PHI in your responses\n"
        "- Use medical terminology appropriately and consistently\n"
        "- Suggest specific improvements rather than vague feedback\n\n"
        "**Context:**\n"
        "Doctor profile and preferences:\n"
        f"{profile_context.strip()}\n\n"
        "Previous conversation:\n"
        f"{history_block}\n\n"
        "Current transcript or note (if provided):\n"
        f"{transcript_block}\n\n"
        "Doctor's question or instruction:\n"
        f"{user_message.strip()}\n\n"
        "**Your Response:**\n"
        "Provide specific, actionable guidance tailored to this doctor's style and needs."
    )
