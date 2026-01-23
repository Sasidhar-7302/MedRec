"""
Generate comprehensive training data for fine-tuning (10-20 hours equivalent).

This generates:
1. Speaker-diarized dialogues with proper HPI/Assessment
2. Training manifests for Whisper fine-tuning
3. Training data for MedLlama HPI/Assessment extraction

Usage:
    python scripts/generate_training_data.py --output-dir data/training_data
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

# Expanded patient profiles
PATIENT_PROFILES = [
    ("54-year-old male", "teacher", "Type 2 diabetes, hypertension"),
    ("62-year-old female", "chef", "Obesity, GERD"),
    ("41-year-old male", "runner", "No significant past medical history"),
    ("70-year-old female", "retired nurse", "Hypertension, osteoporosis"),
    ("33-year-old male", "software engineer", "Anxiety, IBS"),
    ("58-year-old female", "accountant", "Type 2 diabetes, hyperlipidemia"),
    ("45-year-old male", "construction worker", "Smoking history, GERD"),
    ("67-year-old female", "retired teacher", "Hypertension, arthritis"),
    ("39-year-old male", "lawyer", "No significant past medical history"),
    ("72-year-old female", "retired", "Multiple comorbidities"),
]

# Expanded symptom sets with more detail
SYMPTOM_SETS = [
    {
        "symptoms": "three to four semi-formed bowel movements per day with mild urgency and no bleeding",
        "duration": "three weeks",
        "onset": "gradual",
        "history": "pancolitis on vedolizumab for two years",
        "associated": "mild abdominal cramping, no fever",
    },
    {
        "symptoms": "episodic right upper quadrant pain after fatty meals",
        "duration": "six months",
        "onset": "intermittent",
        "history": "history of gallstones, previous ERCP",
        "associated": "nausea, bloating after meals",
    },
    {
        "symptoms": "burning epigastric pain, worse at night and when lying flat",
        "duration": "two months",
        "onset": "progressive",
        "history": "erosive esophagitis on PPI therapy",
        "associated": "regurgitation, occasional dysphagia",
    },
    {
        "symptoms": "unintentional 10 pound weight loss over three months plus intermittent hematochezia",
        "duration": "three months",
        "onset": "gradual",
        "history": "known ulcerative colitis, currently on mesalamine",
        "associated": "fatigue, decreased appetite",
    },
    {
        "symptoms": "persistent bloating and early satiety after meals",
        "duration": "four months",
        "onset": "progressive",
        "history": "suspected gastroparesis, diabetes",
        "associated": "nausea, occasional vomiting",
    },
    {
        "symptoms": "chronic diarrhea with mucus, five to six times daily",
        "duration": "eight months",
        "onset": "gradual",
        "history": "Crohn's disease, small bowel involvement",
        "associated": "abdominal pain, fatigue, joint pain",
    },
    {
        "symptoms": "dysphagia to solids, progressing over six months",
        "duration": "six months",
        "onset": "progressive",
        "history": "GERD, Barrett's esophagus on surveillance",
        "associated": "chest pain, weight loss",
    },
    {
        "symptoms": "alternating constipation and diarrhea with abdominal discomfort",
        "duration": "one year",
        "onset": "chronic",
        "history": "IBS-M, anxiety",
        "associated": "bloating, stress-related symptoms",
    },
    {
        "symptoms": "bright red blood per rectum with bowel movements",
        "duration": "two weeks",
        "onset": "acute",
        "history": "diverticulosis, no previous bleeding",
        "associated": "no pain, normal bowel habits",
    },
    {
        "symptoms": "severe abdominal pain in right lower quadrant",
        "duration": "two days",
        "onset": "acute",
        "history": "no significant GI history",
        "associated": "fever, nausea, decreased appetite",
    },
]

# Expanded assessments with severity
ASSESSMENTS = [
    {
        "diagnosis": "Ulcerative colitis, mild to moderate flare",
        "severity": "mild to moderate",
        "activity": "active",
        "plan": "Continue vedolizumab, add short course budesonide 9mg daily for 8 weeks",
        "rationale": "Symptoms suggest disease activity despite biologic therapy",
    },
    {
        "diagnosis": "Crohn's disease, small bowel involvement, moderate activity",
        "severity": "moderate",
        "activity": "active",
        "plan": "Escalate to ustekinumab, order MR enterography to assess extent",
        "rationale": "Persistent symptoms indicate need for treatment escalation",
    },
    {
        "diagnosis": "GERD with Barrett's esophagus, poorly controlled",
        "severity": "moderate",
        "activity": "active",
        "plan": "Increase PPI to twice daily, schedule surveillance EGD in 6 months",
        "rationale": "Symptoms persist despite current therapy, need for surveillance",
    },
    {
        "diagnosis": "Functional dyspepsia versus IBS-M",
        "severity": "mild",
        "activity": "chronic",
        "plan": "Start low-FODMAP diet, order celiac serologies and H. pylori testing",
        "rationale": "Need to rule out organic causes before functional diagnosis",
    },
    {
        "diagnosis": "Gallstone disease, symptomatic",
        "severity": "moderate",
        "activity": "active",
        "plan": "Refer for laparoscopic cholecystectomy, dietary modifications",
        "rationale": "Symptomatic gallstones require surgical intervention",
    },
    {
        "diagnosis": "Inflammatory bowel disease, likely Crohn's, moderate activity",
        "severity": "moderate",
        "activity": "active",
        "plan": "Start biologic therapy, order colonoscopy with biopsies",
        "rationale": "Persistent symptoms and weight loss indicate active disease",
    },
    {
        "diagnosis": "Esophageal stricture, likely related to GERD",
        "severity": "moderate",
        "activity": "chronic",
        "plan": "Schedule EGD with dilation, optimize PPI therapy",
        "rationale": "Progressive dysphagia requires endoscopic evaluation and treatment",
    },
    {
        "diagnosis": "Irritable bowel syndrome, mixed type",
        "severity": "mild to moderate",
        "activity": "chronic",
        "plan": "Dietary modifications, consider low-dose antidepressant",
        "rationale": "Functional symptoms respond to lifestyle and pharmacologic interventions",
    },
    {
        "diagnosis": "Lower GI bleeding, likely diverticular",
        "severity": "mild",
        "activity": "acute",
        "plan": "Colonoscopy to evaluate source, supportive care",
        "rationale": "Acute bleeding requires endoscopic evaluation",
    },
    {
        "diagnosis": "Acute appendicitis, clinical suspicion",
        "severity": "severe",
        "activity": "acute",
        "plan": "Urgent surgical consultation, CT abdomen if diagnosis uncertain",
        "rationale": "Acute presentation requires immediate evaluation",
    },
]

FOLLOW_UPS = [
    "Follow up in 6 weeks to assess response to therapy.",
    "Call sooner if bleeding recurs, fever develops, or symptoms worsen.",
    "Repeat labs (CBC, CMP, CRP, ESR) prior to next visit.",
    "Schedule colonoscopy in 6 months for surveillance.",
    "Coordinate nutrition consult within 2 weeks.",
    "Return in 4 weeks to assess medication response.",
    "Follow up in 3 months or sooner if symptoms persist.",
    "Schedule procedure within 2 weeks.",
    "Return for urgent evaluation if symptoms worsen.",
    "Follow up after procedure results available.",
]


def build_detailed_case(seed: int) -> Dict:
    """Build a detailed case with proper HPI and Assessment."""
    random.seed(seed)
    patient, occupation, pmh = random.choice(PATIENT_PROFILES)
    symptom_set = random.choice(SYMPTOM_SETS)
    assessment = random.choice(ASSESSMENTS)
    follow_up = random.choice(FOLLOW_UPS)

    # Build HPI from patient statements
    hpi = (
        f"{patient} working as {occupation} presents with {symptom_set['symptoms']} "
        f"for {symptom_set['duration']}. Symptoms began {symptom_set['onset']}. "
        f"Associated symptoms include {symptom_set['associated']}. "
        f"Past medical history includes {pmh}. "
        f"Current history includes {symptom_set['history']}."
    )

    # Build dialogue WITHOUT explicit speaker labels (realistic conversation)
    # In real conversations, speakers don't say "Doctor:" or "Patient:"
    dialogue_lines_no_labels = [
        f"Good morning, thanks for coming in today. What brings you in?",
        f"I've been having {symptom_set['symptoms']} for about {symptom_set['duration']} now.",
        f"Can you tell me more about when this started and how it's been progressing?",
        f"It started {symptom_set['onset']}, and I've also noticed {symptom_set['associated']}.",
        f"Any bleeding, fever, weight loss, or other concerning symptoms?",
        f"No bleeding or fever. Maybe some weight change, but I'm not sure.",
        f"What about your past medical history?",
        f"I have {pmh}, and I'm currently on treatment for {symptom_set['history']}.",
        f"Let me review your labs and we'll make a plan.",
        f"Based on your symptoms and history, my assessment is {assessment['diagnosis']}, "
        f"{assessment['severity']} severity, {assessment['activity']} activity.",
        f"What does that mean for treatment?",
        f"{assessment['plan']}. {assessment['rationale']}",
        f"When should I follow up?",
        f"{follow_up}",
        f"That sounds good. I'll follow up as instructed.",
    ]
    
    # Also create version WITH speaker labels for training supervision
    dialogue_lines_with_labels = [
        f"Doctor: Good morning, thanks for coming in today. What brings you in?",
        f"Patient: I've been having {symptom_set['symptoms']} for about {symptom_set['duration']} now.",
        f"Doctor: Can you tell me more about when this started and how it's been progressing?",
        f"Patient: It started {symptom_set['onset']}, and I've also noticed {symptom_set['associated']}.",
        f"Doctor: Any bleeding, fever, weight loss, or other concerning symptoms?",
        f"Patient: No bleeding or fever. Maybe some weight change, but I'm not sure.",
        f"Doctor: What about your past medical history?",
        f"Patient: I have {pmh}, and I'm currently on treatment for {symptom_set['history']}.",
        f"Doctor: Let me review your labs and we'll make a plan.",
        f"Doctor: Based on your symptoms and history, my assessment is {assessment['diagnosis']}, "
        f"{assessment['severity']} severity, {assessment['activity']} activity.",
        f"Patient: What does that mean for treatment?",
        f"Doctor: {assessment['plan']}. {assessment['rationale']}",
        f"Patient: When should I follow up?",
        f"Doctor: {follow_up}",
        f"Patient: That sounds good. I'll follow up as instructed.",
    ]

    # Use unlabeled version for realistic training (model learns to infer speakers)
    dialogue = "\n".join(dialogue_lines_no_labels)
    dialogue_labeled = "\n".join(dialogue_lines_with_labels)  # For reference/validation

    # Build structured summary
    findings = (
        f"Chief complaint: {symptom_set['symptoms']}. "
        f"Duration: {symptom_set['duration']}. "
        f"Associated symptoms: {symptom_set['associated']}. "
        f"Past medical history: {pmh}. "
        f"Current history: {symptom_set['history']}."
    )

    assessment_text = (
        f"1. {assessment['diagnosis']} - {assessment['severity']} severity, "
        f"{assessment['activity']} activity. {assessment['rationale']}"
    )

    plan_text = f"{assessment['plan']}. {follow_up}"

    summary = (
        f"HPI (History of Present Illness):\n{hpi}\n\n"
        f"Findings:\n{findings}\n\n"
        f"Assessment:\n{assessment_text}\n\n"
        f"Plan:\n{plan_text}\n\n"
        f"Medications/Orders:\n{assessment['plan']}\n\n"
        f"Follow-up:\n{follow_up}"
    )

    # Speaker sequence for reference (alternating doctor/patient)
    speaker_sequence = []
    for i in range(len(dialogue_lines_no_labels)):
        # Pattern: doctor starts, alternates
        speaker = "doctor" if i % 2 == 0 else "patient"
        speaker_sequence.append(speaker)
    
    return {
        "dialogue": dialogue,  # Unlabeled (realistic)
        "dialogue_labeled": dialogue_labeled,  # Labeled (for reference)
        "transcript": dialogue,  # For Whisper training (unlabeled)
        "transcript_labeled": dialogue_labeled,  # Labeled version (for validation)
        "summary": summary,
        "hpi": hpi,
        "findings": findings,
        "assessment": assessment_text,
        "plan": plan_text,
        "speakers": speaker_sequence,  # True speaker sequence for each line
    }


def generate_whisper_manifest(entries: List[Dict], output_path: Path, split: str = "train"):
    """Generate Whisper training manifest."""
    manifest = []
    for idx, entry in enumerate(entries):
        # Include both unlabeled (realistic) and labeled (for training) versions
        manifest.append({
            "id": f"{split}_{idx:05d}",
            "text": entry["transcript"],  # Unlabeled (realistic conversation)
            "text_labeled": entry.get("transcript_labeled", entry["transcript"]),  # Labeled (for supervision)
            "speakers": entry.get("speakers", []),  # True speaker sequence
            "source": "synthetic",
            "note": "Audio file needs to be generated using TTS. Use 'text' for realistic training, 'text_labeled' for supervision.",
        })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Generated Whisper manifest: {output_path} ({len(manifest)} entries)")


def generate_medllama_data(entries: List[Dict], output_path: Path, split: str = "train"):
    """Generate MedLlama training data for HPI/Assessment extraction."""
    training_data = []
    
    for entry in entries:
        # Format for instruction-following model
        # Use unlabeled dialogue (realistic) - model learns to identify speakers from context
        training_data.append({
            "instruction": "Extract HPI (History of Present Illness) and Assessment from this medical conversation transcript. Identify patient statements for HPI and doctor statements for Assessment.",
            "input": entry["dialogue"],  # Unlabeled (realistic)
            "input_labeled": entry.get("dialogue_labeled", entry["dialogue"]),  # Labeled (for reference)
            "output": f"HPI (History of Present Illness):\n{entry['hpi']}\n\nAssessment:\n{entry['assessment']}",
            "hpi": entry["hpi"],
            "assessment": entry["assessment"],
            "speakers": entry.get("speakers", []),  # True speaker sequence for validation
        })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Generated MedLlama training data: {output_path} ({len(training_data)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive training data")
    parser.add_argument(
        "--count",
        type=int,
        default=2000,
        help="Number of training samples to generate (2000 = ~10-20 hours equivalent)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training_data"),
        help="Output directory for training data",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Fraction of data for training (rest for validation)",
    )
    
    args = parser.parse_args()
    
    print(f"Generating {args.count} training samples...")
    print(f"Output directory: {args.output_dir}")
    
    # Generate all cases
    all_entries = []
    for idx in range(args.count):
        case = build_detailed_case(idx)
        all_entries.append(case)
    
    # Split into train/val
    random.seed(42)
    random.shuffle(all_entries)
    split_idx = int(len(all_entries) * args.train_split)
    train_entries = all_entries[:split_idx]
    val_entries = all_entries[split_idx:]
    
    # Save full dataset
    full_output = args.output_dir / "full_dataset.jsonl"
    full_output.parent.mkdir(parents=True, exist_ok=True)
    with open(full_output, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Saved full dataset: {full_output} ({len(all_entries)} entries)")
    
    # Generate Whisper manifests
    generate_whisper_manifest(
        train_entries,
        args.output_dir / "whisper_train.jsonl",
        "train"
    )
    generate_whisper_manifest(
        val_entries,
        args.output_dir / "whisper_val.jsonl",
        "val"
    )
    
    # Generate MedLlama training data
    generate_medllama_data(
        train_entries,
        args.output_dir / "medllama_train.jsonl",
        "train"
    )
    generate_medllama_data(
        val_entries,
        args.output_dir / "medllama_val.jsonl",
        "val"
    )
    
    # Save summary
    summary = {
        "total_samples": len(all_entries),
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "train_split": args.train_split,
        "files_generated": [
            "full_dataset.jsonl",
            "whisper_train.jsonl",
            "whisper_val.jsonl",
            "medllama_train.jsonl",
            "medllama_val.jsonl",
        ],
        "estimated_hours": len(all_entries) * 0.01,  # Rough estimate: ~36 seconds per sample
    }
    
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training Data Generation Complete!")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_entries)}")
    print(f"Train samples: {len(train_entries)}")
    print(f"Validation samples: {len(val_entries)}")
    print(f"Estimated audio hours: {summary['estimated_hours']:.1f} hours")
    print(f"\nFiles generated in: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

