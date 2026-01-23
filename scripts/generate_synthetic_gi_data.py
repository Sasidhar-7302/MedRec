"""
Generate synthetic GI specialist conversations + summaries.

Usage:
    python scripts/generate_synthetic_gi_data.py --count 100 --output data/synthetic_gi_pairs.jsonl

Each JSONL line contains:
{
  "dialogue": "Doctor: ...\\nPatient: ...",
  "summary": "Findings: ...\\nAssessment: ...\\nPlan: ...",
  "hpi": "...",
  "assessment": "...",
  "plan": "..."
}

This script is intentionally lightweight so it can run offline. For higher realism,
replace the template logic with calls to an LLM (OpenAI, Claude, etc.) and keep the
same output schema.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PATIENT_PROFILES = [
    ("54-year-old male", "teacher"),
    ("62-year-old female", "chef"),
    ("41-year-old male", "runner"),
    ("70-year-old female", "retired nurse"),
    ("33-year-old male", "software engineer"),
]

SYMPTOM_SETS = [
    ("three semi-formed bowel movements per day with mild urgency and no bleeding", "pancolitis on vedolizumab"),
    ("episodic RUQ pain after fatty meals", "history of gallstones"),
    ("burning epigastric pain, worse at night", "erosive esophagitis"),
    ("unintentional 10 lb weight loss + intermittent hematochezia", "known ulcerative colitis"),
    ("persistent bloating and early satiety", "suspected gastroparesis"),
]

ASSESSMENTS = [
    ("Ulcerative colitis, mild flare", "Continue vedolizumab, short course budesonide"),
    ("Crohn's disease, small bowel involvement", "Escalate to ustekinumab, order MR enterography"),
    ("GERD with Barrett's esophagus", "Increase PPI to BID, schedule surveillance EGD"),
    ("Functional dyspepsia vs IBS-M", "Start low-FODMAP diet, order celiac serologies"),
    ("Gallstone disease", "Refer for laparoscopic cholecystectomy"),
]

FOLLOW_UPS = [
    "Follow up in 6 weeks to assess symptoms.",
    "Call sooner if bleeding recurs or fever develops.",
    "Repeat labs (CBC, CMP, CRP) prior to next visit.",
    "Schedule colonoscopy in 6 months.",
    "Coordinate nutrition consult within 2 weeks.",
]


def build_case(seed: int) -> dict:
    random.seed(seed)
    patient, occupation = random.choice(PATIENT_PROFILES)
    symptoms, history = random.choice(SYMPTOM_SETS)
    assessment, plan = random.choice(ASSESSMENTS)
    follow = random.choice(FOLLOW_UPS)

    opening = f"Patient is a {patient} working as a {occupation}."
    hpi = (
        f"{opening} Reports {symptoms}. Denies fevers, night sweats, or travel. "
        f"Current history includes {history}."
    )
    assessment_text = assessment
    plan_text = f"{plan}. {follow}"

    dialogue = [
        f"Doctor: Good morning, thanks for coming in. How have your symptoms been?",
        f"Patient: I've noticed {symptoms}.",
        "Doctor: Any bleeding, fever, or weight loss?",
        "Patient: No bleeding or fever. Maybe a slight weight change.",
        f"Doctor: You're still taking {history}, correct?",
        "Patient: Yes, on schedule.",
        "Doctor: Let's review labs and make a plan.",
        f"Doctor: My assessment is {assessment_text}.",
        f"Patient: What does the plan look like?",
        f"Doctor: {plan_text}",
        "Patient: Sounds good. I'll follow up as instructed.",
    ]

    summary = f"Findings: {hpi}\nAssessment: {assessment_text}\nPlan: {plan_text}"

    return {
        "dialogue": "\n".join(dialogue),
        "summary": summary,
        "hpi": hpi,
        "assessment": assessment_text,
        "plan": plan_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic GI dialogue pairs.")
    parser.add_argument("--count", type=int, default=100, help="Number of samples to generate.")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_gi_pairs.jsonl"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for idx in range(args.count):
            case = build_case(idx)
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Wrote {args.count} synthetic pairs to {args.output}")


if __name__ == "__main__":
    main()
