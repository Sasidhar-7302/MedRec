import os
from pathlib import Path

GT_DIR = Path("data/GiTestValid")
files = list(GT_DIR.glob("*.txt"))

for f in files:
    try:
        with open(f, "r", encoding="utf-8") as f_obj:
            content = f_obj.read()
            print(f"{f.name}: Read {len(content)} chars. First 50: {content[:50]}")
    except Exception as e:
        print(f"{f.name}: Failed to read as UTF-8. Error: {e}")
        try:
            with open(f, "r", encoding="utf-16") as f_obj:
                content = f_obj.read()
                print(f"{f.name}: Read as UTF-16 {len(content)} chars. First 50: {content[:50]}")
        except Exception as e2:
            print(f"{f.name}: Failed to read as UTF-16. Error: {e2}")
