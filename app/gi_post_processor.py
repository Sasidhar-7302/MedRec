"""
GI-Specific Post-Processor for Transcription and Summarization
===============================================================
Corrects common transcription errors, normalizes medical terminology,
and ensures consistent formatting of clinical notes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .gi_terms import load_gi_terms


class GIPostProcessor:
    """Post-processor for GI medical text to improve accuracy."""

    # Common transcription errors: (wrong -> correct)
    COMMON_CORRECTIONS: Dict[str, str] = {
        # Medication names
        "humera": "Humira",
        "remicaid": "Remicade",
        "stellara": "Stelara",
        "intyvio": "Entyvio",
        "entivio": "Entyvio",
        "zeljentz": "Xeljanz",
        "zeposia": "Zeposia",
        "sky ritzy": "Skyrizi",
        "skyrizi": "Skyrizi",
        "trivis": "Truvada",
        "lialda": "Lialda",
        "apriso": "Apriso",
        "canasa": "Canasa",
        "rowasa": "Rowasa",
        "u ceris": "Uceris",
        "uceris": "Uceris",
        "viberzi": "Viberzi",
        "linzess": "Linzess",
        "amitiza": "Amitiza",
        "motegrity": "Motegrity",
        "methotrexate": "methotrexate",
        "6 mp": "6-MP",
        "6mp": "6-MP",
        "azathioprine": "azathioprine",
        "mesa lamine": "mesalamine",
        "mezalamine": "mesalamine",
        "budesonide": "budesonide",
        "prednisone": "prednisone",
        
        # Conditions
        "crohns": "Crohn's",
        "crohn's": "Crohn's",
        "crohns disease": "Crohn's disease",
        "uc": "UC",
        "ulcerative colitis": "ulcerative colitis",
        "ibd": "IBD",
        "ibs": "IBS",
        "gerd": "GERD",
        "gastroesophageal reflux": "gastroesophageal reflux",
        "barretts": "Barrett's",
        "barrett's esophagus": "Barrett's esophagus",
        "h pylori": "H. pylori",
        "h. pylori": "H. pylori",
        "helicobacter": "Helicobacter",
        "c diff": "C. diff",
        "c. diff": "C. diff",
        "c difficile": "C. difficile",
        "sibo": "SIBO",
        "s.i.b.o.": "SIBO",
        "maldigestion": "maldigestion",
        "malabsorption": "malabsorption",
        "steatorrhea": "steatorrhea",
        "stay at area": "steatorrhea",
        "eosinophilic": "eosinophilic",
        "eoe": "EoE",
        "e.o.e.": "EoE",
        
        # Procedures
        "egd": "EGD",
        "e.g.d.": "EGD",
        "colonoscopy": "colonoscopy",
        "ercp": "ERCP",
        "e.r.c.p.": "ERCP",
        "mrcp": "MRCP",
        "m.r.c.p.": "MRCP",
        "endoscopy": "endoscopy",
        "esophago gastro duodenoscopy": "esophagogastroduodenoscopy",
        "peg tube": "PEG tube",
        
        # Lab values
        "cbc": "CBC",
        "c.b.c.": "CBC",
        "cmp": "CMP",
        "c.m.p.": "CMP",
        "crp": "CRP",
        "c.r.p.": "CRP",
        "esr": "ESR",
        "e.s.r.": "ESR",
        "alt": "ALT",
        "ast": "AST",
        "lfts": "LFTs",
        "calprotectin": "calprotectin",
        "cal protectin": "calprotectin",
        
        # Dosing
        "bid": "BID",
        "b.i.d.": "BID",
        "tid": "TID",
        "t.i.d.": "TID",
        "qd": "QD",
        "q.d.": "QD",
        "prn": "PRN",
        "p.r.n.": "PRN",
        "mg": "mg",
        "milligrams": "mg",
        
        # Anatomy
        "esophagus": "esophagus",
        "esophageal": "esophageal",
        "jejunum": "jejunum",
        "ileum": "ileum",
        "duodenum": "duodenum",
        "colon": "colon",
        "sigmoid": "sigmoid",
        "cecum": "cecum",
        "rectum": "rectum",
        
        # Common mishearings
        "die area": "diarrhea",
        "diearea": "diarrhea",
        "nausia": "nausea",
        "nausious": "nauseous",
        "eppy gastric": "epigastric",
        "epi gastric": "epigastric",
        "dis fagia": "dysphagia",
        "dis fasia": "dysphagia",
        "odin oh fagia": "odynophagia",
        
        # Missing terms from Kaggle tests
        "post prandial": "postprandial",
        "post-prandial": "postprandial",
        "chole cystitis": "cholecystitis",
        "cholecystitis": "cholecystitis",
        "appendicitis": "appendicitis",
        "gastro enteritis": "gastroenteritis",
        "gastroenteritis": "gastroenteritis",
        "billiary": "biliary",
        "biliary colic": "biliary colic",
        "rlq": "RLQ",
        "r.l.q.": "RLQ",
        "ruq": "RUQ",
        "r.u.q.": "RUQ",
        "llq": "LLQ",
        "l.l.q.": "LLQ",
        "luq": "LUQ",
        "l.u.q.": "LUQ",
        
        # Biologics/Advanced Therapies
        "simponi": "Simponi",
        "rinvoq": "Rinvoq",
        "omvoh": "Omvoh",
        "velsipity": "Velsipity",
        "skyrizi": "Skyrizi",
        "entyvio": "Entyvio",
        "remicade": "Remicade",
        "humira": "Humira",
        "simponi aria": "Simponi Aria",
        "inflectra": "Inflectra",
        "avsola": "Avsola",
        "renflexis": "Renflexis",

        # Colloquial to Medical
        "after eating": "postprandial",
        "after meals": "postprandial",
        "stomach ache": "abdominal pain",
        "belly pain": "abdominal pain",
        "tummy pain": "abdominal pain",
        "throw up": "vomit",
        "throwing up": "vomiting",
        "puked": "vomited",
        "poop": "stool",
        "pain when peeing": "dysuria",
        "pain with urination": "dysuria",
        "peeing a lot": "urinary frequency",
        "urinating a lot": "urinary frequency",
        "hard to swallow": "dysphagia",
        "painful swallowing": "odynophagia",
    }

    # Patterns for normalization
    DOSAGE_PATTERN = re.compile(r"(\d+)\s*(mg|mcg|ml|g)\b", re.IGNORECASE)
    FREQUENCY_PATTERN = re.compile(r"\b(once|twice|three times)\s+(a\s+)?(day|daily|weekly)", re.IGNORECASE)

    def __init__(self):
        self.gi_vocabulary = set(term.lower() for term in load_gi_terms())
        self._build_correction_map()

    def _build_correction_map(self):
        """Build case-insensitive correction map."""
        self.correction_map = {}
        for wrong, correct in self.COMMON_CORRECTIONS.items():
            self.correction_map[wrong.lower()] = correct

    def process_transcription(self, text: str) -> str:
        """Process transcription to correct common errors."""
        if not text:
            return text

        result = text
        
        # Apply word-level corrections
        result = self._apply_word_corrections(result)
        
        # Normalize dosages
        result = self._normalize_dosages(result)
        
        # Fix capitalization for GI terms
        result = self._fix_gi_capitalization(result)
        
        # Clean up whitespace
        result = self._clean_whitespace(result)
        
        return result

    def _apply_word_corrections(self, text: str) -> str:
        """Apply common word corrections with word boundary protection."""
        result = text
        
        # Sort by length (longest first) to avoid partial matches
        sorted_corrections = sorted(
            self.correction_map.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        for wrong, correct in sorted_corrections:
            # Case-insensitive replacement with word boundaries (\b)
            # This prevents 'past' becoming 'pAST' (matching 'ast')
            pattern = re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)
            result = pattern.sub(correct, result)
        
        return result

    def _normalize_dosages(self, text: str) -> str:
        """Normalize medication dosages (e.g., '20 mg' -> '20mg')."""
        def normalize(match):
            number = match.group(1)
            unit = match.group(2).lower()
            return f"{number}{unit}"
        
        return self.DOSAGE_PATTERN.sub(normalize, text)

    def _fix_gi_capitalization(self, text: str) -> str:
        """Fix capitalization for known GI terms using regex to preserve whitespace."""
        terms = {"egd", "ercp", "mrcp", "ibd", "ibs", "gerd", "ppi", 
                 "cbc", "cmp", "crp", "esr", "lfts", "alt", "ast",
                 "bid", "tid", "qd", "prn", "nafld", "nash", "hcc", "rlq", "ruq", "llq", "luq"}
        
        # Sort by length (longest first)
        sorted_terms = sorted(list(terms), key=len, reverse=True)
        
        result = text
        for term in sorted_terms:
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            result = pattern.sub(term.upper(), result)
        
        return result

    def _clean_whitespace(self, text: str) -> str:
        """Clean up extra whitespace."""
        # Collapse multiple spaces
        text = re.sub(r" +", " ", text)
        # Fix spacing around punctuation
        text = re.sub(r" ([.,;:!?])", r"\1", text)
        return text.strip()

    def process_summary(self, summary: str) -> str:
        """Process generated summary for consistency without breaking formatting."""
        if not summary:
            return summary

        result = summary
        
        # Apply word corrections and capitalizations ONLY
        # Do NOT use _clean_whitespace or full process_transcription as they might break layout
        result = self._apply_word_corrections(result)
        result = self._normalize_dosages(result)
        result = self._fix_gi_capitalization(result)
        
        # Ensure proper section formatting
        result = self._format_sections(result)
        
        return result

    def _format_sections(self, text: str) -> str:
        """Ensure proper formatting of clinical note sections."""
        # Standard section headers
        sections = [
            "HPI", "History of Present Illness",
            "Findings", "Assessment", "Plan",
            "Medications", "Orders", "Follow-up"
        ]
        
        for section in sections:
            # Add colon if missing
            pattern = re.compile(rf"^({section})\s*(?!:)", re.MULTILINE | re.IGNORECASE)
            text = pattern.sub(r"\1:", text)
        
        return text

    def validate_gi_terminology(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Validate GI terminology in text.
        Returns (valid_terms, potentially_misspelled_terms)
        """
        # Extract all potential medical terms (capitalized words, acronyms, etc.)
        words = set(re.findall(r"\b[A-Za-z][a-z]*(?:'s)?\b", text))
        
        valid_terms = []
        suspicious_terms = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.gi_vocabulary:
                valid_terms.append(word)
            elif len(word) > 5 and self._looks_medical(word):
                # Might be a misspelled medical term
                closest = self._find_closest_term(word_lower)
                if closest:
                    suspicious_terms.append(f"{word} (did you mean '{closest}'?)")
        
        return valid_terms, suspicious_terms

    def _looks_medical(self, word: str) -> bool:
        """Heuristic to detect if a word looks like medical terminology."""
        medical_suffixes = ["itis", "osis", "ectomy", "oscopy", "emia", "pathy", 
                          "plasty", "tomy", "gram", "scopy"]
        medical_prefixes = ["gastro", "hepato", "colon", "esophag", "duoden", 
                           "jejun", "ile", "entero"]
        
        word_lower = word.lower()
        return any(word_lower.endswith(s) for s in medical_suffixes) or \
               any(word_lower.startswith(p) for p in medical_prefixes)

    def _find_closest_term(self, word: str, threshold: int = 2) -> Optional[str]:
        """Find closest GI term using Levenshtein distance."""
        best_match = None
        best_distance = threshold + 1
        
        for term in self.gi_vocabulary:
            if abs(len(term) - len(word)) > threshold:
                continue
            
            distance = self._levenshtein(word, term)
            if distance < best_distance:
                best_distance = distance
                best_match = term
        
        return best_match if best_distance <= threshold else None

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance."""
        if len(s1) < len(s2):
            return GIPostProcessor._levenshtein(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


# Module-level instance for convenience
_post_processor: Optional[GIPostProcessor] = None


def get_post_processor() -> GIPostProcessor:
    """Get or create the post-processor instance."""
    global _post_processor
    if _post_processor is None:
        _post_processor = GIPostProcessor()
    return _post_processor


def process_transcription(text: str) -> str:
    """Convenience function to process transcription."""
    return get_post_processor().process_transcription(text)


def process_summary(text: str) -> str:
    """Convenience function to process summary."""
    return get_post_processor().process_summary(text)
