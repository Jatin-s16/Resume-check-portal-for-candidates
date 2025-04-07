import streamlit as st

from main import rate_resume_against_jd


st.title("Check your resume compatibility with any job description")

resume_pdf = st.file_uploader("Upload Resume (PDF)", type="pdf")
jd_pdf = st.file_uploader("Upload Job Description (PDF)", type="pdf")

if resume_pdf and jd_pdf:
    score = rate_resume_against_jd(resume_pdf, jd_pdf)
    st.success(f"Compatibility Score: {score} / 10")