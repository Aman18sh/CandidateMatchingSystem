import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from src.functions import (
    embedding_model, load_pdf_file, filter_to_minimal_docs, 
    resume_features_extraction, resume_to_text, jobpost_feature_extraction, job_post_to_text
)
from src.utils import clean_text
from src.prompts import resume_prompt, job_post_prompt, candidate_matching_prompt
from embed import building_vectordb

from rank_bm25 import BM25Okapi
import tempfile
import os
import shutil
import base64
import io

# ------------------ UI CONFIGURATION ------------------ 

st.set_page_config(
    page_title="AI Candidate Matching",
    page_icon="",
    layout="wide"
)

st.title("AI-Driven Candidate Matching System")
st.markdown("Intelligent ranking, explainability, and hybrid retrieval in one place.")


st.sidebar.header("Configuration")
top_k = st.sidebar.slider("Top K Candidates", 1, 10, 5)
temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.7)


# ------------------ INPUT AREA -------------------------- 

st.subheader("Job Description")
job_description_input = st.text_area(
    "Enter Job Description",
    height=200,
    placeholder="Paste the job description here..."
)

uploaded_files = st.file_uploader(
    "Upload Candidate Resumes (PDF only)",
    type=["pdf"],
    accept_multiple_files=True
)

run_btn = st.button("Run Candidate Matching")



# ------------------MAIN PIPELINE FOR CANDIDATE SCREENING STARTS FROM HERE--------------------------


if run_btn:

    # Validation
    if not job_description_input or not uploaded_files:
        st.error("Please upload resumes and enter a job description.")
        st.stop()

    with st.spinner("Processing resumes..."):

        # Setting up the LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temperature,
            google_api_key=os.environ["GEMINI_API_KEY"]
        ) 
        parser = JsonOutputParser() # for structured json response

        # creating a temporary directory as load_pdf_file fn expects a directory containing resumes
        temp_dir = tempfile.mkdtemp()
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

        extracted_resume = load_pdf_file(temp_dir)
        filtered_resume = filter_to_minimal_docs(extracted_resume)
        resume_extraction = resume_features_extraction(
            filtered_resume, llm, clean_text, resume_prompt, resume_to_text, parser
        )

        doc_search = building_vectordb(resume_extraction,embedding_model)

        job_description_doc = jobpost_feature_extraction(
            job_description_input, llm, clean_text, job_post_prompt, job_post_to_text, parser
        )
        job_description = job_description_doc.page_content


    # Retrieving Candidate profile through hybrid search as per the jobpost or job description 

    with st.spinner("Performing hybrid retrieval..."):

        # Dense Retrievel (Semantic Search)
        retriever = doc_search.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
        dense_results = retriever.invoke(job_description)

        # Sparse Retrievel (Keyword Search)
        docs = [doc.page_content for doc in resume_extraction]
        tokenized_docs = [d.lower().split() for d in docs]
        bm25 = BM25Okapi(tokenized_docs)

        # Calculating the matching score for each candidate profile as per job description
        keywords = job_description.lower().split()
        bm25_scores = bm25.get_scores(keywords)

        # Taking the highest score candidate profile
        top_ids = bm25_scores.argsort()[-top_k:][::-1]
        sparse_results = [resume_extraction[i] for i in top_ids]

        metadata_filtered = [
            c for c in resume_extraction
            if c.metadata["experience"] >= job_description_doc.metadata["experience_required"]
        ]

        # Combining Retrieval Results from all, by taking intersection using id
        combined = dense_results + sparse_results + metadata_filtered
        unique_docs = list({doc.metadata["id"]: doc for doc in combined}.values())


    # Candidate Matching and Ranking using candidate_matching_prompt

    with st.spinner("Ranking candidates using AI..."):

        candidate_context = "\n\n".join(doc.page_content for doc in unique_docs)
        chain = candidate_matching_prompt | llm

        response_text = chain.invoke({
            "candidate_context": candidate_context,
            "job_description": job_description
        })

    st.success("Candidate ranking completed!")


    # -----------------------------DISPLAYING RESULT ON THE UI------------------------------------

  

    st.subheader("Final Ranked Candidates")
    # st.text(response_text)

    # Assume the LLM output is plain text w/ structured sections
    # candidates_output = response_text.split("\n\n")
    response_text_str = response_text.content if hasattr(response_text, "content") else str(response_text)

    candidates_output = response_text_str.split("\n\n")


    # Collapsible Cards 
    for i, candidate_block in enumerate(candidates_output):

        if not candidate_block.strip():
            continue

        # Extract name for the title
        first_line = candidate_block.split("\n")[0]
        candidate_name = first_line.replace("**", "").strip()

        with st.expander(f"👤 {candidate_name}", expanded=False):
            st.markdown(candidate_block)

        
            # Extracting score
            if "Score:" in candidate_block:
                try:
                    score_val = float(candidate_block.split("Score:")[1].split("\n")[0].strip())
                    st.progress(score_val / 100)  # convert to 0–1
                except:
                    pass

            # Download individuals report button
            if st.button(
            f"Download Report for {candidate_name}",
            key=f"download_btn_{i}"
            ):
                buffer = io.BytesIO()
                pdf = SimpleDocTemplate(buffer)
                styles = getSampleStyleSheet()
                story = [Paragraph(candidate_block.replace("\n", "<br/>"), styles["Normal"])]
                pdf.build(story)

                encoded_pdf = base64.b64encode(buffer.getvalue()).decode()
                st.markdown(
                    f'<a href="data:application/pdf;base64,{encoded_pdf}" download="{candidate_name}_report.pdf">Download PDF</a>',
                    unsafe_allow_html=True
                )
