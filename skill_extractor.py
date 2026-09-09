"""
Skill & Keyword Extraction Module
Identifies known technical skills/keywords present in a document
using a predefined skill dictionary (extendable via data/skills_list.csv).
"""

import csv
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_FILE = os.path.join(BASE_DIR, "data", "skills_list.csv")


def load_skills(path: str = SKILLS_FILE) -> list:
    skills = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                skills.append(row[0].strip().lower())
    return skills


def extract_skills(text: str, skills_vocab: list = None) -> list:
    """Return the list of known skills found in the given text."""
    if skills_vocab is None:
        skills_vocab = load_skills()

    text_lower = text.lower()
    found = set()
    for skill in skills_vocab:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)
    return sorted(found)


if __name__ == "__main__":
    sample = "Proficient in Python, SQL, Machine Learning, Pandas and Flask."
    print(extract_skills(sample))
