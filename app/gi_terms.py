"""Utilities for supplying GI-specific vocabulary to prompts and post-processing."""

from __future__ import annotations

from pathlib import Path
from typing import List


DATA_FILE = Path("data/gi_terms.txt")
DEFAULT_TERMS: List[str] = [
    "pancolitis", "Crohn's disease", "ulcerative colitis", "Barrett's esophagus", "GERD",
    "bile duct stricture", "vedolizumab", "ustekinumab", "adalimumab", "calprotectin",
    "fibroscan", "ERCP", "MRCP", "EGD", "colonoscopy", "gastroscopy", "proctosigmoidoscopy",
    "cholecystectomy", "esophagogastroduodenoscopy", "esophagitis", "duodenitis", "gastritis",
    "diverticulitis", "diverticulosis", "hemorrhoids", "fissure", "fistula", "ascites",
    "cirrhosis", "hepatitis", "pancreatitis", "biliary", "stent", "biopsy", "polypectomy",
    "dysphagia", "odynophagia", "heartburn", "acid reflux", "bloating", "flatulence",
    "constipation", "diarrhea", "hematochezia", "melena", "tenesmus", "urgency", "vomiting", "vomited", "nauseous",
    "abdominal pain", "epigastric", "right lower quadrant", "left lower quadrant", "perianal",
    "mesalamine", "azathioprine", "mercaptopurine", "infliximab", "certolizumab", "golimumab",
    "tofacitinib", "upadacitinib", "risankizumab", "corticosteroids", "prednisone", "budesonide",
    "pantoprazole", "omeprazole", "famotidine", "linaclotide", "lubiprostone", "plecanatide",
    "rifaximin", "cholestyramine", "loperamide", "ondansetron", "promethazine"
]


def load_gi_terms() -> List[str]:
    """Return GI vocabulary from data/gi_terms.txt or defaults."""
    if DATA_FILE.exists():
        terms = [line.strip() for line in DATA_FILE.read_text(encoding="utf-8").splitlines()]
        return [term for term in terms if term]
    return DEFAULT_TERMS


def build_gi_hint(max_terms: int = 80) -> str:
    """Return a comma-separated hint string for prompts."""
    terms = load_gi_terms()[:max_terms]
    if not terms:
        return "gastroenterology terminology"
    return ", ".join(terms)
