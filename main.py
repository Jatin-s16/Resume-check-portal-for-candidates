#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fitz 
import spacy
import re

nlp = spacy.load("en_core_web_sm")


# ## Extracting text from both resume and job description pdf

# In[2]:


def extract_text_from_pdf(pdf_file):
    try:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            return " ".join([page.get_text() for page in doc])
    except Exception as e:
        return f"Error reading PDF: {e}"


# In[3]:


def preprocess(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces/newlines
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)


# In[4]:


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


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rate_resume_against_jd(resume_file, jd_file):
    # Extract and preprocess
    resume_text = extract_text_from_pdf(resume_file)
    jd_text = extract_text_from_pdf(jd_file)

    resume_clean = preprocess(resume_text)
    jd_clean = preprocess(jd_text)

    # TF-IDF vectorization + similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd_clean, resume_clean])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

    # Optional: boost based on structured sections
    resume_sections = extract_sections(resume_text)
    boost = 0

    for section in ["skills", "experience", "education"]:
        section_text = preprocess(resume_sections.get(section, ""))
        if section_text:
            vecs = vectorizer.transform([jd_clean, section_text])
            score = cosine_similarity(vecs[0], vecs[1])[0][0]
            boost += score * 0.1  # small weighted bonus
    
    # Final score (scaled out of 10)
    final_score = min((similarity + boost) * 10, 10)
    return round(final_score, 2)

