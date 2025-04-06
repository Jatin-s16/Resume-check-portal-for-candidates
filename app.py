import streamlit as st

import nbimporter
from main import extract_resume_text, extract_jd_text, calculate_similarity_score, rate_resume_against_jd


st.title("Resume vs Job Description Matcher 🔍")

resume_pdf = st.file_uploader("Upload Resume (PDF)", type="pdf")
jd_pdf = st.file_uploader("Upload Job Description (PDF)", type="pdf")

if resume_pdf and jd_pdf:
    score = rate_resume_against_jd(resume_pdf, jd_pdf)
    st.success(f"Compatibility Score: {score} / 10")