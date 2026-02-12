"""
Professional Performance & Accuracy Testing for GI Scribe
==========================================================
This script tests the summarization pipeline using realistic GI conversations.
It measures:
1. Performance (latency)
2. Summary completeness (does it include HPI, Assessment, Plan?)
3. Key term extraction accuracy
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime

# Set CUDA library path before imports
os.environ["PATH"] = str(Path(__file__).parent / "cuda_libs") + os.pathsep + os.environ.get("PATH", "")

from app.config import AppConfig
from app.two_pass_summarizer import TwoPassSummarizer

# Test cases: Selected diverse GI scenarios from synthetic data
TEST_CASES = [
    {
        "name": "UC Mild Flare",
        "dialogue": """Doctor: Good morning, thanks for coming in. How have your symptoms been?
Patient: I've noticed three semi-formed bowel movements per day with mild urgency and no bleeding.
Doctor: Any bleeding, fever, or weight loss?
Patient: No bleeding or fever. Maybe a slight weight change.
Doctor: You're still taking pancolitis on vedolizumab, correct?
Patient: Yes, on schedule.
Doctor: Let's review labs and make a plan.
Doctor: My assessment is Ulcerative colitis, mild flare.
Patient: What does the plan look like?
Doctor: Continue vedolizumab, short course budesonide. Repeat labs (CBC, CMP, CRP) prior to next visit.
Patient: Sounds good. I'll follow up as instructed.""",
        "expected_assessment": "Ulcerative colitis, mild flare",
        "expected_plan_keywords": ["vedolizumab", "budesonide", "labs"],
    },
    {
        "name": "GERD/Barrett's",
        "dialogue": """Doctor: Good morning, thanks for coming in. How have your symptoms been?
Patient: I've noticed burning epigastric pain, worse at night.
Doctor: Any bleeding, fever, or weight loss?
Patient: No bleeding or fever. Maybe a slight weight change.
Doctor: You're still taking erosive esophagitis, correct?
Patient: Yes, on schedule.
Doctor: Let's review labs and make a plan.
Doctor: My assessment is GERD with Barrett's esophagus.
Patient: What does the plan look like?
Doctor: Increase PPI to BID, schedule surveillance EGD. Coordinate nutrition consult within 2 weeks.
Patient: Sounds good. I'll follow up as instructed.""",
        "expected_assessment": "GERD with Barrett's esophagus",
        "expected_plan_keywords": ["PPI", "EGD", "nutrition"],
    },
    {
        "name": "Crohn's Disease",
        "dialogue": """Doctor: Good morning, thanks for coming in. How have your symptoms been?
Patient: I've noticed persistent bloating and early satiety.
Doctor: Any bleeding, fever, or weight loss?
Patient: No bleeding or fever. Maybe a slight weight change.
Doctor: You're still taking suspected gastroparesis, correct?
Patient: Yes, on schedule.
Doctor: Let's review labs and make a plan.
Doctor: My assessment is Crohn's disease, small bowel involvement.
Patient: What does the plan look like?
Doctor: Escalate to ustekinumab, order MR enterography. Call sooner if bleeding recurs or fever develops.
Patient: Sounds good. I'll follow up as instructed.""",
        "expected_assessment": "Crohn's disease",
        "expected_plan_keywords": ["ustekinumab", "MR enterography"],
    },
    {
        "name": "Gallstone Disease",
        "dialogue": """Doctor: Good morning, thanks for coming in. How have your symptoms been?
Patient: I've noticed episodic RUQ pain after fatty meals.
Doctor: Any bleeding, fever, or weight loss?
Patient: No bleeding or fever. Maybe a slight weight change.
Doctor: You're still taking history of gallstones, correct?
Patient: Yes, on schedule.
Doctor: Let's review labs and make a plan.
Doctor: My assessment is Gallstone disease.
Patient: What does the plan look like?
Doctor: Refer for laparoscopic cholecystectomy. Coordinate nutrition consult within 2 weeks.
Patient: Sounds good. I'll follow up as instructed.""",
        "expected_assessment": "Gallstone disease",
        "expected_plan_keywords": ["cholecystectomy", "laparoscopic"],
    },
    {
        "name": "IBS/Functional Dyspepsia",
        "dialogue": """Doctor: Good morning, thanks for coming in. How have your symptoms been?
Patient: I've noticed persistent bloating and early satiety.
Doctor: Any bleeding, fever, or weight loss?
Patient: No bleeding or fever. Maybe a slight weight change.
Doctor: You're still taking suspected gastroparesis, correct?
Patient: Yes, on schedule.
Doctor: Let's review labs and make a plan.
Doctor: My assessment is Functional dyspepsia vs IBS-M.
Patient: What does the plan look like?
Doctor: Start low-FODMAP diet, order celiac serologies. Schedule colonoscopy in 6 months.
Patient: Sounds good. I'll follow up as instructed.""",
        "expected_assessment": "Functional dyspepsia",
        "expected_plan_keywords": ["FODMAP", "celiac", "colonoscopy"],
    },
]


def check_section_present(summary: str, section: str) -> bool:
    """Check if a section header is present in the summary."""
    section_lower = section.lower()
    return any(s in summary.lower() for s in [section_lower + ":", section_lower + " -", f"**{section_lower}**"])


def check_keywords_present(summary: str, keywords: list) -> tuple:
    """Check which keywords are present in the summary."""
    found = []
    missing = []
    for kw in keywords:
        if kw.lower() in summary.lower():
            found.append(kw)
        else:
            missing.append(kw)
    return found, missing


def run_tests():
    print("=" * 70)
    print("GI SCRIBE - PROFESSIONAL PERFORMANCE & ACCURACY TESTING")
    print("=" * 70)
    print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load config
    config = AppConfig.load(Path("config.json"))
    summarizer = TwoPassSummarizer(config.summarizer)

    results = []
    total_time = 0

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n[Test {i}/{len(TEST_CASES)}] {test['name']}")
        print("-" * 50)

        # Run summarization
        try:
            result = summarizer.summarize(test["dialogue"])
            summary = summarizer._format_structured_summary(result)
            elapsed = result.runtime_s
            total_time += elapsed
            success = True
        except Exception as e:
            summary = f"ERROR: {e}"
            elapsed = 0
            success = False

        # Evaluate results
        has_findings = check_section_present(summary, "Findings") or check_section_present(summary, "HPI")
        has_assessment = check_section_present(summary, "Assessment")
        has_plan = check_section_present(summary, "Plan")

        assessment_match = test["expected_assessment"].lower() in summary.lower()
        found_kw, missing_kw = check_keywords_present(summary, test["expected_plan_keywords"])
        keyword_score = len(found_kw) / len(test["expected_plan_keywords"]) * 100 if test["expected_plan_keywords"] else 100

        result_data = {
            "name": test["name"],
            "success": success,
            "time_s": elapsed,
            "has_findings": has_findings,
            "has_assessment": has_assessment,
            "has_plan": has_plan,
            "assessment_match": assessment_match,
            "keyword_score": keyword_score,
            "missing_keywords": missing_kw,
            "summary_length": len(summary),
        }
        results.append(result_data)

        # Print results
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Structure: Findings={has_findings}, Assessment={has_assessment}, Plan={has_plan}")
        print(f"  Assessment Match: {assessment_match}")
        print(f"  Keyword Score: {keyword_score:.0f}% (Missing: {missing_kw or 'None'})")
        print(f"  Summary Length: {len(summary)} chars")
        if not success:
            print(f"  ERROR: {summary}")

    # Summary Statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    successful = [r for r in results if r["success"]]
    if successful:
        avg_time = sum(r["time_s"] for r in successful) / len(successful)
        avg_keyword = sum(r["keyword_score"] for r in successful) / len(successful)
        structure_score = sum(1 for r in successful if r["has_findings"] and r["has_assessment"] and r["has_plan"]) / len(successful) * 100
        assessment_score = sum(1 for r in successful if r["assessment_match"]) / len(successful) * 100

        print(f"Tests Run: {len(TEST_CASES)}")
        print(f"Tests Passed: {len(successful)}/{len(TEST_CASES)}")
        print()
        print("PERFORMANCE:")
        print(f"  Average Latency: {avg_time:.2f}s")
        print(f"  Total Time: {total_time:.2f}s")
        print()
        print("ACCURACY:")
        print(f"  Structure Completeness: {structure_score:.0f}% (Findings+Assessment+Plan present)")
        print(f"  Assessment Accuracy: {assessment_score:.0f}%")
        print(f"  Plan Keyword Accuracy: {avg_keyword:.0f}%")
        print()
        print("OVERALL SCORE:", end=" ")
        overall = (structure_score + assessment_score + avg_keyword) / 3
        if overall >= 90:
            print(f"{overall:.0f}% - EXCELLENT [PASS]")
        elif overall >= 75:
            print(f"{overall:.0f}% - GOOD")
        elif overall >= 50:
            print(f"{overall:.0f}% - NEEDS IMPROVEMENT")
        else:
            print(f"{overall:.0f}% - POOR")
    else:
        print("All tests failed!")

    print("\n" + "=" * 70)
    print(f"Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_tests()
