
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import AppConfig
from app.two_pass_summarizer import TwoPassSummarizer

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_config():
    import json
    with open("config.json", "r") as f:
        return AppConfig.from_dict(json.load(f))

def main():
    setup_logging()
    logger = logging.getLogger("test_rag_pipeline")
    
    logger.info("Loading config...")
    config = load_config()
    
    logger.info("Initializing Summarizer...")
    summarizer = TwoPassSummarizer(config.summarizer)
    
    # Transcript from GAS0001.mp3 (approximate from previous logs)
    # This transcript mimics the structure expected after mapping
    transcript = """
[00:00:00 - 00:00:03] Doctor: What brings you in today?
[00:00:03 - 00:00:07] Patient: I've just been feeling, like, very nauseated.
[00:00:10 - 00:00:12] Patient: It feels like all the time right now.
[00:00:14 - 00:00:16] Doctor: Hmm. When did this start?
[00:00:19 - 00:00:22] Patient: It's been over a week. Maybe not quite two weeks, but around nine days.
[00:00:30 - 00:00:35] Doctor: Do you have any pain?
[00:00:35 - 00:00:40] Patient: Sometimes a little bit of burning in my chest, like heartburn. And acid coming up.
[00:00:41 - 00:00:45] Doctor: Okay. Does it happen after eating?
[00:00:46 - 00:00:50] Patient: Yeah, mostly after dinner.
[00:01:00 - 00:01:10] Doctor: It sounds like GERD. I think we should try some omeprazole.
[00:01:11 - 00:01:15] Doctor: Take 40mg daily before breakfast.
[00:01:20 - 00:01:25] Doctor: Also, any family history of colon cancer?
[00:01:26 - 00:01:30] Patient: My dad had it when he was 50.
[00:01:31 - 00:01:35] Doctor: Okay, we should schedule a colonoscopy for screening then.
    """
    
    logger.info("Running summarization on hardcoded transcript...")
    try:
        summary = summarizer.summarize_text(transcript)
        print("\n=== GENERATED SUMMARY ===\n")
        print(summary)
        print("\n=========================\n")
    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
