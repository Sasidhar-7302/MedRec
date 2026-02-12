
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.guideline_rag import GuidelineRAG

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_query(rag, query):
    print(f"\nQUERY: '{query}'")
    results = rag.retrieve(query, k=1)
    if not results:
        print("  -> No results found.")
    else:
        for r in results:
            print(f"  -> MATCH: {r['topic']}")
            print(f"     Content: {r['content'][:100]}...")

def main():
    setup_logging()
    print("Initializing GuidelineRAG...")
    try:
        rag = GuidelineRAG()
    except Exception as e:
        print(f"FAILED to initialize RAG: {e}")
        return

    # Test Cases
    queries = [
        "Patient has heartburn and acid regurgitation after meals.",
        "Bloody diarrhea and urgency with left-sided abdominal pain.",
        "Family history of colon cancer in father at age 50.",
        "Follow up for Barrett's esophagus without dysplasia."
    ]

    for q in queries:
        test_query(rag, q)

if __name__ == "__main__":
    main()
