"""
Realistic Medical Conversation Testing - Kaggle Dataset
=========================================================
Tests GI Scribe with realistic doctor-patient conversations from Kaggle.
These are more diverse and natural than synthetic data.
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime

os.environ["PATH"] = str(Path(__file__).parent / "cuda_libs") + os.pathsep + os.environ.get("PATH", "")

from app.config import AppConfig
from app.two_pass_summarizer import TwoPassSummarizer

# Realistic Kaggle conversations
KAGGLE_CASES = [
    {
        "name": "Potential Pregnancy (Nausea/Vomiting/Missed Period)",
        "dialogue": """D: What brings you in today?
P: I've just been feeling like very nauseated for, it feels like all the time right now.
D: When did this start?
P: It's been over a week. Maybe not quite two weeks, but like around then. Yeah, maybe like 9 days.
D: And um, is it, do you always have the sensation of nausea or is it related, or does it come and go?
P: Oh, I think it's like worse when I am smelling something really bad, and it's worse in the morning. But I feel like it's always kind of there.
D: I see OK. Um, have you had any vomiting?
P: Uh, yeah, like um, yeah, like a lot of days I'll throw up like once or twice.
D: And this has all been over the past week, week two weeks or so?
P: Yeah.
D: OK, um, any other symptoms that you have?
P: Oh no, just, well, like I feel like I've I've had to pee a lot more. Um, but I don't, I don't think that's like related.
D: How often do you have to pee?
P: Probably every probably every like 2, maybe, maybe every hour I get certain times in a day.
D: Do you wake up at night to pee?
P: Oh, like it doesn't wake me up but it's like I wake up because I hear something. Then I'll be like, oh I think I should probably go to the bathroom.
D: Do you have any pain when you're peeing?
P: No.
D: Have you had any other stomach related issues? Any belly pain or changes in your bowel movements?
P: Ah no, I don't think so, well my stomach has been like it feels like a little crampy. I thought that maybe it could be my period but like I'm not on my period.
D: When did you last have your period?
P: Oh well, let me think. Um, like six weeks ago.
D: Is it normal for you to go six weeks without a period?
P: Oh, come to think of it, I don't, I don't think so.
D: Any fevers?
P: No.
D: Are you currently sexually active?
P: Yeah just like with my boyfriend.
D: And do you use protection while having sex?
P: Yeah, we we just use condoms. I used to be on birth control but it always made me feel like like kind of sick so I don't use that anymore.""",
        "expected_keywords": [["nausea", "nauseated"], "vomiting", ["missed period", "amenorrhea", "no period"], "pregnancy", ["urinary frequency", "polyuria", "peeing a lot", "urinating frequently"]],
    },
    {
        "name": "Biliary Colic / Gallbladder (RUQ Pain)",
        "dialogue": """D: I was wondering if you could tell me a little bit about what brought you in to the Emergency Department today?
P: Yeah, so nice to meet you. I've been having this pain right in my abdomen. It's kind of like in the upper right area.
D: OK, and so uh, when, where is this painting located exactly?
P: So it's just in the upper right corner of my abdomen, right below where the lungs are, and it, yeah, it's just I have this severe pain that's going on.
D: OK, and how long is it been going on for?
P: So it's been going on for the last few days and it got worse today.
D: OK, and does the pain radiate anywhere?
P: Uh no, it stays right in the in the spot that I told you right in the right upper corner.
D: OK, and when did the pain start? Or if you could tell me what were you doing right prior to the pain starting?
P: So I think it started after just three days ago after I had a meal like I I think it was after lunch around half an hour or an hour after lunch.
D: How would you describe the character or the quality of the pain?
P: So it's like a sharp, I would describe it as like a sharp pain.
D: OK, and on a scale of 1 to 10, 10 being the most severe pain, what would you rate it as?
P: I would rate it as, right now I would rate it as an 8.
D: OK, and has there been anything that you've tried to make this pain any better?
P: I tried taking just like Advil and Tylenol, but it didn't really seem to help the pain too much.
D: OK, and have you had any other associated symptoms such as nausea or or vomiting?
P: I've I've had some nausea over the past few days, but I haven't vomited anything.
D: Do you take any medications on a regular basis?
P: Sometimes I take like some antacids when I get heartburn.
D: OK, and any family history of gallbladder disease or cardiovascular disease in the family?
P: Um, so my father died of a stroke when he was in his 60s, my mother does have gallstones.
D: OK, and do you drink alcohol?
P: Uh, yeah sometimes, maybe one or two glasses of wine every night.""",
        "expected_keywords": [["RUQ pain", "Right Upper Quadrant", "Right Upper Abdomen"], "gallbladder", "biliary", "nausea", ["postprandial", "after meals", "following meals", "after eating"]],
    },
    {
        "name": "Acute Gastroenteritis (Diarrhea/Vomiting)",
        "dialogue": """D: I understand that you've been having some diarrhea. Can you tell me a little bit about that?
P: Yeah, sure I had, I've been having diarrhea for the past three to four days, and it's been pretty bad. I couldn't go, uh I couldn't go to my classes, had to skip because it was just it was just embarrassing.
D: OK, I'm sorry to hear that. Have you ever experienced something like this before?
P: Um, have I experienced something like this before? Uhm no, I don't think so.
D: How many times a day have you been having episodes of diarrhea?
P: Oh my god, I have I've lost count. I'm going every every hour.
D: OK, every hour. OK, and have you noticed any blood in your stool?
P: Um not really. I don't think so.
D: OK, um have you noticed any pain associated with it?
P: Well, I didn't start off with pain, but I I I do have a bit of a cramp now since yesterday.
D: Before this happened, is there anything that you've done differently that you've been eating differently?
P: Um, I I did go to um, I did go to a new restaurant 5 days ago.
D: OK, what did you have there?
P: Um I I just got some rice with chicken Manchurian, it was a nice little Chinese restaurant.
D: In terms of the last three to four days, have you experienced any nausea or vomiting?
P: Yeah yeah, I have actually. I I did vomit, I think twice yesterday.
D: Have you noticed any blood in the vomit?
P: No, it's just watery.
D: Have you noticed any fever or chills in the last few days?
P: Um, I've just been feeling really tired.
D: Do you have any chronic conditions?
P: I have asthma.
D: Do you take any medication?
P: Yeah, I've got some puffers that my family doc gave me.""",
        "expected_keywords": ["diarrhea", "gastroenteritis", "vomiting", ["abdominal cramps", "cramping", "pain", "crampy"], "food"],
    },
    {
        "name": "Pediatric Viral Gastroenteritis (6yo)",
        "dialogue": """D: Would you mind by starting with what brought you in today?
P: Yeah, so I'm just coming in with my son. He's six years old and yeah, just over the last few days he's not been himself and he's been having this stomach ache. Uh, so it started around like I would say 3 days ago and then he's also having vomiting for the last two days, he's vomited in total six times over the last two days and then yesterday he also developed a fever as well. I managed to measure it and it was 38.3 degrees Celsius.
D: OK, could you describe the vomit?
P: Yes, so vomit like, it started two days ago. The first day it was just like he puked up the things that he had eaten, but yesterday it was it was mainly just like uh just yellowish material.
D: OK, have you noticed any bile or blood in the vomit?
P: Uh no I didn't notice any blood and I didn't notice any green material.
D: And with regards to his diarrhea, has there been any any blood or or any color changes?
P: No, no, I didn't notice any blood at all. It's definitely just very very watery and he's had to go probably even just in the last day, probably around like six or seven times.
D: OK, has he had any headaches at all?
P: No, not that I know.
D: Any fever or chills?
P: Yeah, so that prompted me to take his temperature yesterday, we had to wrap him up with like more than two blankets and he was still feeling cold.
D: Has there been anybody in the house been sick?
P: Uh no, no one has been sick recently, however he does, they did let them go back into school at one point, so I don't know if he caught something from school.
D: And immunizations are up to date?
P: Yeah, immunizations all up to date.""",
        "expected_keywords": ["pediatric", "vomiting", "diarrhea", "fever", "viral", "gastroenteritis"],
    },
    {
        "name": "Suspected Appendicitis (RLQ Pain)",
        "dialogue": """D: So I understand you have been experiencing some abdominal pain?
P: Yeah yeah stomach hurt, started hurting more last couple of days, maybe 3 days ago I think.
D: OK, so for three days. Did you have pain before that?
P: It felt weird, like crampy. I just, I just thought I was constipated, 'cause I've been haven't been able to be able to go to the bathroom as well.
D: Can you tell me kind of where you're feeling the pain the most?
P: Yeah, kinda like near my right hip like lower where my stomach is that kind of right and below my belly button.
D: OK, have you ever had pain like this in the past?
P: No, never.
D: Is there anything that you can think of this made it feel better?
P: Honestly, just resting flat makes it feel a little bit better, but nothing much.
D: OK, is there anything that makes it worse?
P: Touching it. Also I puked the the other day and that made it definitely feel worse, just that whole contraction in my body was nasty.
D: OK, was it just the one time?
P: Yeah.
D: Have you noticed any fevers lately?
P: Felt a bit hot the other day, didn't take a temperature though.
D: Have you noticed any changes to your bowel habits?
P: Yeah. Before it was fairly regular, about once a day. Five days ago, I started getting constipated, last time I went to the bathroom was two days ago.
D: Do you have any chronic conditions?
P: I got diabetes, type 2.
D: And you take anything for your diabetes?
P: Metformin.
D: Still have your appendix?
P: Yeah, yeah, I never had that problem as a kid.""",
        "expected_keywords": ["appendicitis", ["RLQ pain", "Right Lower Quadrant", "Right Lower Abdomen"], "nausea", "constipation", "rebound"],
    },
]


def run_kaggle_tests():
    print("=" * 70)
    print("GI SCRIBE - KAGGLE REALISTIC CONVERSATION TESTING")
    print("=" * 70)
    print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Cases: {len(KAGGLE_CASES)}")
    print()

    config = AppConfig.load(Path("config.json"))
    summarizer = TwoPassSummarizer(config.summarizer)

    results = []
    total_time = 0

    for i, test in enumerate(KAGGLE_CASES, 1):
        print(f"\n{'='*70}")
        print(f"[Test {i}/{len(KAGGLE_CASES)}] {test['name']}")
        print("=" * 70)

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

        # Check for expected keywords
        found = []
        missing = []
        for kw_group in test["expected_keywords"]:
            # If it's a list, treat as synonyms (any match is good)
            synonyms = kw_group if isinstance(kw_group, list) else [kw_group]
            
            match_found = False
            for synonym in synonyms:
                if synonym.lower() in summary.lower():
                    match_found = True
                    found.append(synonyms[0]) # Record the primary term as found
                    break
            
            if not match_found:
                missing.append(synonyms[0])

        keyword_score = len(found) / len(test["expected_keywords"]) * 100 if test["expected_keywords"] else 100

        result_data = {
            "name": test["name"],
            "success": success,
            "time_s": elapsed,
            "keyword_score": keyword_score,
            "missing_keywords": missing,
            "summary_length": len(summary),
        }
        results.append(result_data)

        print(f"Time: {elapsed:.2f}s")
        print(f"Keywords Found: {keyword_score:.0f}% (Missing: {missing or 'None'})")
        print(f"Summary Length: {len(summary)} chars")
        print()
        print("GENERATED SUMMARY:")
        print("-" * 50)
        print(summary)

    # Summary Statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    successful = [r for r in results if r["success"]]
    if successful:
        avg_time = sum(r["time_s"] for r in successful) / len(successful)
        avg_keyword = sum(r["keyword_score"] for r in successful) / len(successful)

        print(f"Tests Run: {len(KAGGLE_CASES)}")
        print(f"Tests Passed: {len(successful)}/{len(KAGGLE_CASES)}")
        print()
        print("PERFORMANCE:")
        print(f"  Average Latency: {avg_time:.2f}s")
        print(f"  Total Time: {total_time:.2f}s")
        print()
        print("ACCURACY:")
        print(f"  Average Keyword Match: {avg_keyword:.0f}%")

    print("\n" + "=" * 70)
    print(f"Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_kaggle_tests()
