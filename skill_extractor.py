import logging
import re
from typing import List, Tuple
import spacy

logger = logging.getLogger(__name__)

# Load small English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

# Comprehensive skill dictionary with categories
SKILL_DATABASE = {
    "Backend": [
        "Python", "Flask", "Django", "FastAPI", "Spring", "Java",
        "Node.js", "Express", "Go", "Rust", "C++", "C#", ".NET"
    ],
    "Frontend": [
        "React", "Vue", "Angular", "JavaScript", "TypeScript", "HTML",
        "CSS", "Tailwind", "Next.js", "Svelte"
    ],
    "Data": [
        "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch",
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
        "Data Analysis", "Statistics", "R", "SAS"
    ],
    "DevOps": [
        "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Jenkins",
        "CI/CD", "Terraform", "Linux", "Git"
    ],
    "Databases": [
        "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Cassandra",
        "Elasticsearch", "DynamoDB"
    ],
    "Other": [
        "REST APIs", "GraphQL", "Microservices", "SOLID", "OOP",
        "Design Patterns", "Agile", "System Design"
    ]
}

COMMON_SKILLS = []
for skills in SKILL_DATABASE.values():
    COMMON_SKILLS.extend(skills)

# Regex patterns for common acronyms/technical terms
TECH_PATTERNS = {
    r"\bML\b": "Machine Learning",
    r"\bDL\b": "Deep Learning",
    r"\bNLP\b": "NLP",
    r"\bAPI\b": "REST APIs",
    r"\bRESTful\b": "REST APIs",
    r"\bOOP\b": "OOP",
}

def extract_skills(text: str) -> Tuple[List[str], List[Tuple[str, float]]]:
    """
    Extract skills from input text using multiple strategies.

    Args:
        text: CV or LinkedIn text

    Returns:
        Tuple of (extracted_skills, confidence_scores)
        confidence_scores is list of (skill, confidence) tuples
    """
    if not text or not text.strip():
        return [], []

    extracted = set()
    confidence_scores = {}

    try:
        # Strategy 1: Direct substring matching (high confidence)
        text_lower = text.lower()
        for skill in COMMON_SKILLS:
            if skill.lower() in text_lower:
                extracted.add(skill)
                confidence_scores[skill] = 0.95
    except Exception as e:
        logger.error(f"Error in substring matching: {e}")

    try:
        # Strategy 2: Regex patterns for acronyms
        for pattern, skill in TECH_PATTERNS.items():
            if re.search(pattern, text):
                extracted.add(skill)
                confidence_scores[skill] = 0.85
    except Exception as e:
        logger.error(f"Error in regex matching: {e}")

    try:
        # Strategy 3: spaCy NER for entities
        if nlp:
            doc = nlp(text)
            # Extract PERSON entities (often skill names), ORG (frameworks/tools)
            for ent in doc.ents:
                if ent.label_ in ["PRODUCT", "ORG"]:
                    for skill in COMMON_SKILLS:
                        if skill.lower() in ent.text.lower():
                            extracted.add(skill)
                            confidence_scores[skill] = 0.80
    except Exception as e:
        logger.error(f"Error in spaCy NER: {e}")

    # Build confidence-scored list, sorted by confidence descending
    results = [(skill, confidence_scores.get(skill, 0.75)) for skill in extracted]
    results.sort(key=lambda x: x[1], reverse=True)

    return [r[0] for r in results], results

def get_skill_category(skill: str) -> str:
    """Return the category of a skill."""
    for category, skills in SKILL_DATABASE.items():
        if skill in skills:
            return category
    return "Other"

# Unit tests
def test_extract_skills():
    """Basic tests for skill extraction."""
    test_cases = [
        ("I use Python and Flask daily", {"Python", "Flask"}),
        ("Machine Learning with TensorFlow and PyTorch", {"Machine Learning", "TensorFlow", "PyTorch"}),
        ("", set()),
        ("No skills here!@#$", set()),
    ]

    for text, expected in test_cases:
        extracted, _ = extract_skills(text)
        assert expected.issubset(set(extracted)), f"Failed for: {text}"

    logger.info("All tests passed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_extract_skills()