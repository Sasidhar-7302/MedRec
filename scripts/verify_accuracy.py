
import json
import logging
import time
from pathlib import Path
from typing import Dict, List
import re

from app.config import load_config
from app.summarizer import OllamaSummarizer

def parse_summary(summary_text: str) -> Dict[str, str]:
    """Parse summary into sections."""
    sections = {}
    current_section = None
    buffer = []
    
    for line in summary_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers
        lower_line = line.lower()
        if lower_line.startswith('findings:'):
            if current_section:
                sections[current_section] = '\n'.join(buffer).strip()
            current_section = 'hpi' # Map Findings to hpi for comparison
            buffer = [line.split(':', 1)[1].strip()]
        elif lower_line.startswith('assessment:'):
            if current_section:
                sections[current_section] = '\n'.join(buffer).strip()
            current_section = 'assessment'
            buffer = [line.split(':', 1)[1].strip()]
        elif lower_line.startswith('plan:'):
            if current_section:
                sections[current_section] = '\n'.join(buffer).strip()
            current_section = 'plan'
            buffer = [line.split(':', 1)[1].strip()]
        else:
            if current_section:
                buffer.append(line)
                
    if current_section:
        sections[current_section] = '\n'.join(buffer).strip()
        
    return sections

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple overlap similarity."""
    if not text1 or not text2:
        return 0.0
        
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    if not set1 or not set2:
        return 0.0
        
    intersection = set1.intersection(set2)
    return len(intersection) / len(set1.union(set2))

def verify_accuracy():
    print("Loading config...")
    config = load_config()
    summarizer = OllamaSummarizer(config.summarizer)
    
    if not summarizer.health_check():
        print("FATAL: Ollama service not running or model unavailable.")
        return

    print("Loading validation data...")
    val_file = Path("data/synthetic_gi_pairs_val.jsonl")
    if not val_file.exists():
        print(f"FATAL: Validation file {val_file} not found.")
        return

    data = []
    with open(val_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Limit to subset for quick verification if needed, or run all
    data = data[:2] 
    
    print(f"Running verification on {len(data)} samples...")
    
    results = {
        'total': 0,
        'hpi_similarity': 0.0,
        'assessment_similarity': 0.0,
        'plan_similarity': 0.0,
        'perfect_format': 0
    }
    
    start_time = time.time()
    
    for i, item in enumerate(data):
        print(f"Processing {i+1}/{len(data)}...")
        input_text = item['dialogue']
        expected_hpi = item.get('hpi', '')
        expected_assessment = item.get('assessment', '')
        expected_plan = item.get('plan', '')
        
        try:
            # Generate summary
            result = summarizer.summarize(input_text, style="Narrative")
            parsed = parse_summary(result.summary)
            
            # metrics
            hpi_score = calculate_similarity(parsed.get('hpi', ''), expected_hpi)
            assess_score = calculate_similarity(parsed.get('assessment', ''), expected_assessment)
            plan_score = calculate_similarity(parsed.get('plan', ''), expected_plan)
            
            results['hpi_similarity'] += hpi_score
            results['assessment_similarity'] += assess_score
            results['plan_similarity'] += plan_score
            results['total'] += 1
            
            is_perfect = (hpi_score > 0.8 and assess_score > 0.8 and plan_score > 0.8)
            if is_perfect:
                results['perfect_format'] += 1
                
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            
    avg_hpi = results['hpi_similarity'] / results['total'] if results['total'] > 0 else 0
    avg_assess = results['assessment_similarity'] / results['total'] if results['total'] > 0 else 0
    avg_plan = results['plan_similarity'] / results['total'] if results['total'] > 0 else 0
    accuracy = (avg_hpi + avg_assess + avg_plan) / 3 * 100
    
    total_time = time.time() - start_time
    
    print("\n" + "="*50)
    print("ACCURACY VERIFICATION REPORT")
    print("="*50)
    print(f"Samples: {results['total']}")
    print(f"Time: {total_time:.2f}s ({total_time/results['total']:.2f}s/sample)")
    print(f"HPI Similarity: {avg_hpi:.2%}")
    print(f"Assessment Similarity: {avg_assess:.2%}")
    print(f"Plan Similarity: {avg_plan:.2%}")
    print(f"Overall Accuracy Score: {accuracy:.2f}%")
    print("="*50)
    
    # Save to file
    with open("VERIFICATION_REPORT.md", "w") as f:
        f.write("# Verification Report\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    verify_accuracy()
