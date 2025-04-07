import fitz 
import spacy
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")  # semantic embeddings



## Extracting text from both resume and job description pdf

def extract_text_from_pdf(pdf_file):
    try:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            return " ".join([page.get_text() for page in doc])
    except Exception as e:
        return f"Error reading PDF: {e}"


def preprocess(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces/newlines
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)


def extract_sections(text):
    sections = {"skills": [], "experience": [], "education": []}
    current_section = None

    for line in text.splitlines():
        line_lower = line.strip().lower()
        if "skill" in line_lower:
            current_section = "skills"
        elif "experience" in line_lower:
            current_section = "experience"
        elif "education" in line_lower:
            current_section = "education"
        elif current_section:
            sections[current_section].append(line.strip())

    return {k: " ".join(v) for k, v in sections.items()}


def get_similarity(text1, text2):
    emb1 = model.encode([text1])[0]
    emb2 = model.encode([text2])[0]
    return cosine_similarity([emb1], [emb2])[0][0]



def rate_resume_against_jd(resume_file, jd_file):
    # Extract and preprocess
    resume_text = extract_text_from_pdf(resume_file)
    jd_text = extract_text_from_pdf(jd_file)

    if "Error" in resume_text or "Error" in jd_text:
        return 0.0

    resume_clean = preprocess(resume_text)
    jd_clean = preprocess(jd_text)

    # Get full-text similarity
    base_score = get_similarity(jd_clean, resume_clean)

    # Extract sections
    sections = extract_sections(resume_text)
    weights = {"skills": 0.4, "experience": 0.3, "education": 0.2}
    boost = 0

    for section, weight in weights.items():
        section_text = preprocess(sections.get(section, ""))
        if section_text:
            score = get_similarity(jd_clean, section_text)
            boost += score * weight

    final_score = min((base_score * 0.5 + boost) * 10, 10)
    return round(final_score, 2)