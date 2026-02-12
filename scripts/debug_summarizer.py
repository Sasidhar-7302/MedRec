from app.config import AppConfig
from app.two_pass_summarizer import TwoPassSummarizer, EXTRACTION_PROMPT, STRUCTURING_PROMPT, TEMPLATE_STYLE
from app.gi_post_processor import process_summary
from pathlib import Path

# Sample dialogue from Kaggle test 5 (Appendicitis)
DIALOGUE = """D: So I understand you have been experiencing some abdominal pain?
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
P: Yeah, yeah, I never had that problem as a kid."""

def debug():
    print("Loading config...")
    config = AppConfig.load(Path("config.json"))
    summarizer = TwoPassSummarizer(config.summarizer)

    with open("debug_output.txt", "w", encoding="utf-8") as f:
        try:
            # Pass 1
            f.write("--- Pass 1: Extraction ---\n")
            extraction_prompt = EXTRACTION_PROMPT.format(
                gi_hints=summarizer.gi_hints,
                transcript=DIALOGUE.strip()
            )
            # We can't easily call _invoke_model from here unless we access protected member
            extracted = summarizer._invoke_model(extraction_prompt, temperature=0.1)
            f.write("Raw Extraction Output:\n")
            f.write(extracted + "\n")
            f.write("-" * 50 + "\n")
            
            # Pass 2
            f.write("--- Pass 2: Structuring ---\n")
            structuring_prompt = STRUCTURING_PROMPT.format(
                few_shot_examples=TEMPLATE_STYLE,
                extracted_info=extracted
            )
            f.write(f"Structuring Prompt Length: {len(structuring_prompt)}\n")
            
            structured = summarizer._invoke_model(structuring_prompt, temperature=0.05)
            f.write("Raw Structured Output:\n")
            f.write(structured + "\n")
            f.write("-" * 50 + "\n")
            
            # Post-process
            f.write("--- Post-processing ---\n")
            final = summarizer._enforce_structure(process_summary(structured))
            f.write("Final Summary:\n")
            f.write(final + "\n")
            
        except Exception as e:
            f.write(f"Error: {e}\n")

if __name__ == "__main__":
    debug()
