"""
Two-Pass Summarizer Benchmark
=============================
Tests the two-pass summarization approach for 90%+ accuracy.
"""

import os
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List

# Set CUDA library path before imports
os.environ["PATH"] = str(Path(__file__).parent / "cuda_libs") + os.pathsep + os.environ.get("PATH", "")

from app.config import AppConfig
from app.two_pass_summarizer import TwoPassSummarizer
from app.gi_terms import load_gi_terms


# Test cases for benchmark
TEST_CASES = [
    {
        "id": "uc_flare",
        "name": "Ulcerative Colitis Mild Flare",
        "dialogue": """Doctor: Good morning, thanks for coming in. How have your symptoms been?
Patient: I've been having three to four semi-formed bowel movements per day with mild urgency. No bleeding though.
Doctor: When did this start?
Patient: About three weeks ago. It's been gradual.
Doctor: Any abdominal pain or fever?
Patient: Just mild cramping, no fever.
Doctor: You're on vedolizumab for your pancolitis, correct?
Patient: Yes, I've been on it for two years now.
Doctor: Let me check your labs. Your CRP is slightly elevated at 8. I think this is a mild flare.
Patient: What's the plan?
Doctor: We'll continue the vedolizumab and add a short course of budesonide, 9mg daily for 8 weeks.
Patient: Sounds good.
Doctor: Get your CBC, CMP, and CRP repeated before your next visit in 6 weeks.""",
        "expected_assessment": "ulcerative colitis",
        "expected_plan_keywords": ["vedolizumab", "budesonide"],
    },
    {
        "id": "crohns_escalation",
        "name": "Crohn's Disease Treatment Escalation",
        "dialogue": """Doctor: How are you doing today?
Patient: Not well. The diarrhea has been really bad - about 6 times a day with blood.
Doctor: I'm sorry to hear that. Any weight loss?
Patient: I've lost about 10 pounds in the last month.
Doctor: You've been on infliximab for how long?
Patient: About two years now.
Doctor: Your fecal calprotectin came back at 800, which is quite elevated. I'm concerned about loss of response to infliximab.
Patient: What do we do now?
Doctor: We need to check your drug levels and antibodies. If the levels are low or you have antibodies, we'll switch to ustekinumab.
Patient: Okay.
Doctor: I also want to order an MR enterography to see the extent of disease in your small bowel.""",
        "expected_assessment": "Crohn's",
        "expected_plan_keywords": ["ustekinumab", "MR enterography"],
    },
    {
        "id": "gerd_barretts",
        "name": "GERD with Barrett's Esophagus",
        "dialogue": """Doctor: You're here today for your Barrett's surveillance results.
Patient: Yes, I've been anxious about it.
Doctor: Good news - the biopsies showed Barrett's without dysplasia.
Patient: That's a relief!
Doctor: How have you been doing with your symptoms?
Patient: Much better. The pantoprazole twice a day really helps.
Doctor: Any heartburn or difficulty swallowing?
Patient: None at all.
Doctor: Great. We'll continue the pantoprazole 40mg twice daily. Your next surveillance EGD should be in 3 years.
Patient: Thank you, doctor.""",
        "expected_assessment": "Barrett's",
        "expected_plan_keywords": ["pantoprazole", "EGD"],
    },
    {
        "id": "gallstones",
        "name": "Symptomatic Gallstones",
        "dialogue": """Doctor: What brings you in today?
Patient: I've been having this pain in my right upper belly after I eat, especially fatty foods.
Doctor: How long does the pain last?
Patient: Usually about an hour or two, then it goes away.
Doctor: Any fever, nausea, or vomiting?
Patient: Some nausea, but no fever or vomiting.
Doctor: I reviewed your ultrasound - it shows multiple gallstones.
Patient: What does that mean for me?
Doctor: These are symptomatic gallstones, also called biliary colic. The best treatment is surgery to remove the gallbladder.
Patient: Surgery?
Doctor: Yes, laparoscopic cholecystectomy. It's minimally invasive. I'll refer you to general surgery. In the meantime, follow a low-fat diet.
Patient: Okay, I understand.""",
        "expected_assessment": "gallstone",
        "expected_plan_keywords": ["cholecystectomy", "surgery"],
    },
    {
        "id": "ibs_m",
        "name": "IBS Mixed Type",
        "dialogue": """Doctor: Tell me about what's been going on.
Patient: I've had alternating constipation and diarrhea for about a year now. Lots of bloating too.
Doctor: Does stress make it worse?
Patient: Definitely. When I'm stressed at work, it gets much worse.
Doctor: Any blood in your stool, weight loss, or fever?
Patient: No, none of that.
Doctor: Any family history of colon cancer or inflammatory bowel disease?
Patient: No.
Doctor: Given your symptoms and age, this sounds like irritable bowel syndrome, mixed type.
Patient: What can we do about it?
Doctor: I'd like you to try a low-FODMAP diet. We should also rule out celiac disease with blood tests.
Patient: Okay.
Doctor: If diet changes don't help enough, we can consider a low-dose antidepressant which can help with gut motility.""",
        "expected_assessment": "IBS",
        "expected_plan_keywords": ["FODMAP", "celiac"],
    },
]


def check_section(text: str, section: str) -> bool:
    """Check if a section header is present."""
    patterns = [
        f"{section}:",
        f"{section} -",
        f"**{section}**",
        f"**{section}:",
        f"{section} (History",  # For HPI with full name
    ]
    text_lower = text.lower()
    return any(p.lower() in text_lower for p in patterns)


def run_two_pass_benchmark():
    print("=" * 70)
    print("GI SCRIBE - TWO-PASS SUMMARIZER BENCHMARK")
    print("=" * 70)
    print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Cases: {len(TEST_CASES)}")
    print()

    # Load configuration
    config = AppConfig.load(Path("config.json"))
    summarizer = TwoPassSummarizer(config.summarizer)
    gi_vocabulary = load_gi_terms()

    results = []
    total_time = 0

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n{'='*70}")
        print(f"[Test {i}/{len(TEST_CASES)}] {test['name']}")
        print("=" * 70)

        # Summarize
        print("Pass 1: Extracting clinical information...")
        print("Pass 2: Structuring into clinical note...")
        
        start = time.perf_counter()
        try:
            summary_text = summarizer.summarize_text(test["dialogue"])
            elapsed = time.perf_counter() - start
            total_time += elapsed
            success = True
        except Exception as e:
            print(f"  ERROR: {e}")
            summary_text = ""
            elapsed = 0
            success = False
            continue

        # Analyze results
        has_hpi = check_section(summary_text, "HPI")
        has_findings = check_section(summary_text, "Findings")
        has_assessment = check_section(summary_text, "Assessment")
        has_plan = check_section(summary_text, "Plan")
        has_medications = check_section(summary_text, "Medications")
        has_followup = check_section(summary_text, "Follow-up")

        # Calculate structure score
        core_score = sum([has_hpi, has_assessment, has_plan]) / 3.0
        optional_score = sum([has_findings, has_medications, has_followup]) / 3.0
        structure_score = (core_score * 0.7 + optional_score * 0.3) * 100

        # Check content
        assessment_found = test["expected_assessment"].lower() in summary_text.lower()
        plan_keywords_found = sum(1 for kw in test["expected_plan_keywords"] if kw.lower() in summary_text.lower())
        plan_keyword_score = plan_keywords_found / len(test["expected_plan_keywords"]) * 100

        results.append({
            "name": test["name"],
            "structure_score": structure_score,
            "has_hpi": has_hpi,
            "has_assessment": has_assessment,
            "has_plan": has_plan,
            "assessment_found": assessment_found,
            "plan_keyword_score": plan_keyword_score,
            "time": elapsed,
        })

        # Print results
        print(f"Time: {elapsed:.2f}s (Pass 1 + Pass 2)")
        print(f"Structure: HPI={has_hpi}, Findings={has_findings}, Assessment={has_assessment}, Plan={has_plan}")
        print(f"Structure Score: {structure_score:.0f}%")
        print(f"Assessment Match: {assessment_found}")
        print(f"Plan Keywords: {plan_keyword_score:.0f}%")
        print(f"\nSUMMARY:")
        print("-" * 50)
        print(summary_text[:1000] + ("..." if len(summary_text) > 1000 else ""))

    # Summary Statistics
    print("\n" + "=" * 70)
    print("TWO-PASS SUMMARIZER BENCHMARK RESULTS")
    print("=" * 70)

    if results:
        avg_structure = sum(r["structure_score"] for r in results) / len(results)
        hpi_rate = sum(1 for r in results if r["has_hpi"]) / len(results) * 100
        assessment_rate = sum(1 for r in results if r["has_assessment"]) / len(results) * 100
        plan_rate = sum(1 for r in results if r["has_plan"]) / len(results) * 100
        assessment_match_rate = sum(1 for r in results if r["assessment_found"]) / len(results) * 100
        avg_plan_keywords = sum(r["plan_keyword_score"] for r in results) / len(results)
        avg_time = sum(r["time"] for r in results) / len(results)

        print(f"\nTESTS COMPLETED: {len(results)}/{len(TEST_CASES)}")
        print()
        print("STRUCTURE ACCURACY:")
        print(f"  Average Structure Score: {avg_structure:.0f}%")
        print(f"  HPI Present: {hpi_rate:.0f}%")
        print(f"  Assessment Present: {assessment_rate:.0f}%")
        print(f"  Plan Present: {plan_rate:.0f}%")
        print()
        print("CLINICAL ACCURACY:")
        print(f"  Assessment Match: {assessment_match_rate:.0f}%")
        print(f"  Plan Keywords: {avg_plan_keywords:.0f}%")
        print()
        print("PERFORMANCE:")
        print(f"  Average Time: {avg_time:.2f}s per case")
        print(f"  Total Time: {total_time:.2f}s")
        print()
        
        # Overall score (weighted: structure 50%, clinical 50%)
        clinical_score = (assessment_match_rate + avg_plan_keywords) / 2
        overall_score = (avg_structure * 0.5) + (clinical_score * 0.5)
        
        print("=" * 70)
        print(f"OVERALL SCORE: {overall_score:.0f}%", end=" ")
        if overall_score >= 95:
            print("- EXCELLENT [PASS]")
        elif overall_score >= 90:
            print("- VERY GOOD [PASS]")
        elif overall_score >= 85:
            print("- GOOD")
        elif overall_score >= 80:
            print("- ACCEPTABLE")
        else:
            print("- NEEDS IMPROVEMENT")
        print("=" * 70)

    print(f"\nBenchmark Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == "__main__":
    run_two_pass_benchmark()
