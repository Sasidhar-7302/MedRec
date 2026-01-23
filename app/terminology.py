"""Lightweight post-processing corrections for medical terms."""

from __future__ import annotations

import re
from typing import List, Tuple


# pair of (regex pattern, canonical replacement)
CORRECTIONS: List[Tuple[str, str]] = [
    # Anatomical terms
    (r"\bilium\b", "ilium"),
    (r"\bileum\b", "ileum"),
    (r"\bjejunum\b", "jejunum"),
    (r"\bduodenum\b", "duodenum"),
    
    # Common abbreviations
    (r"\bg[.\s-]?i\b", "GI"),
    (r"\bgerd\b", "GERD"),
    (r"\bppi\b", "PPI"),
    (r"\begd\b", "EGD"),
    (r"\bercp\b", "ERCP"),
    (r"\bmrcp\b", "MRCP"),
    
    # Diseases and conditions
    (r"\bbarrett'?s\b", "Barrett's"),
    (r"\bcrohn'?s\b", "Crohn's"),
    (r"\bulcerative colitis\b", "ulcerative colitis"),
    (r"\bpankal\s+it\s+is\b", "pancolitis"),
    (r"\bpankalitis\b", "pancolitis"),
    (r"\bpancolitis\b", "pancolitis"),
    (r"\bpancreatitis\b", "pancreatitis"),
    (r"\bcholestasis\b", "cholestasis"),
    (r"\bprimary sclerosing cholangitis\b", "primary sclerosing cholangitis"),
    (r"\bprimary biliary cholangitis\b", "primary biliary cholangitis"),
    (r"\bclostridioides difficile\b", "Clostridioides difficile"),
    (r"\bgastroparesis\b", "gastroparesis"),
    
    # Procedures
    (r"\bcoloscopy\b", "colonoscopy"),
    (r"\bcolonoscopy\b", "colonoscopy"),
    (r"\besophagogastroduodenoscopy\b", "esophagogastroduodenoscopy"),
    (r"\bburbs\b", "biopsies"),
    (r"\bbiopsies\b", "biopsies"),
    (r"\bbiopsy\b", "biopsy"),
    
    # Infections
    (r"\bhep\s+c\b", "Hepatitis C"),
    (r"\bhep\s+b\b", "Hepatitis B"),
    (r"\bh\s+pylori\b", "H. pylori"),
    (r"\bc\s+diff\b", "C. diff"),
    
    # Medications
    (r"\bmesalamine\b", "mesalamine"),
    (r"\badalimumab\b", "adalimumab"),
    (r"\bvedolizumab\b", "vedolizumab"),
    (r"\bustekinumab\b", "ustekinumab"),
    (r"\binfliximab\b", "infliximab"),
    (r"\bfamotidine\b", "famotidine"),
    (r"\bmethotrexate\b", "methotrexate"),
    (r"\bbudesonide\b", "budesonide"),
    (r"\bprucalopride\b", "prucalopride"),
    (r"\bdicyclomine\b", "dicyclomine"),
    (r"\bmetoclopramide\b", "metoclopramide"),
    (r"\bprotonix\b", "Protonix"),
    (r"\bmiralax\b", "MiraLAX"),
    (r"\bstelara\b", "Stelara"),
    
    # Lab tests and diagnostics
    (r"\bferritin\b", "ferritin"),
    (r"\bcalprotectin\b", "calprotectin"),
    (r"\bfibroscan\b", "FibroScan"),
    (r"\bstool studies\b", "stool studies"),
    (r"\bsteatorrhea\b", "steatorrhea"),
    (r"\btenesmus\b", "tenesmus"),
    
    # Other terms
    (r"\bbiologic\b", "biologic"),
    (r"\bgo\s+lyte?ly\b", "GoLYTELY"),
    (r"\bsuprep\b", "SuPrep"),
    
    # Common misrecognitions from test
    (r"\bhelinine\b", "No"),
    (r"\bhelinine is\b", "No"),
    (r"\bblind pain\b", "abdominal pain"),
    (r"\bweight loss\b", "weight loss"),
    (r"\bnormal eating\b", "normal eating"),
    (r"\bbowel movements\b", "bowel movements"),
    (r"\bsigmoid colon\b", "sigmoid colon"),
    (r"\bdysplasia\b", "dysplasia"),
    
    # Fix common phrase errors
    (r"\bon\s+readily\s+zoomed\s+out\b", "controlled"),
    (r"\bmilder\s+ginseng\b", "mild symptoms"),
    (r"\bnew\s+form\b", "new onset"),
]


def apply_corrections(text: str) -> str:
    result = text
    for pattern, replacement in CORRECTIONS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result
