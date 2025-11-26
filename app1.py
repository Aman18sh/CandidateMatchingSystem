import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser

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
# import shutil


# ----------------------------------------- UI Interface -----------------------------------------


st.title("AI Candidate Matching System")

job_description_input = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Candidate Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

run_btn = st.button("Run Candidate Matching")


# ----------------------------------------- Pipeline Working -----------------------------------------

if run_btn:
    if not job_description_input or not uploaded_files:
        st.error("Please upload resumes and enter a job description.")
        st.stop()

    st.info("Processing resumes... Please wait.")

    # Initializing LLM 
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=os.environ["GEMINI_API_KEY"]
    ) 
    parser = JsonOutputParser()

    # Extracting Resumes Features 

    # Creating a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Save uploaded PDFs into the temporary directory
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

    # Extracting resume
    extracted_resume = load_pdf_file(temp_dir)
    
    filtered_resume = filter_to_minimal_docs(extracted_resume)
    resume_extraction = resume_features_extraction(filtered_resume, llm, clean_text, resume_prompt, resume_to_text, parser)

    # Building Vector DB 
    doc_search = building_vectordb(resume_extraction, embedding_model)

    # Job Description Extraction
    job_description_doc = jobpost_feature_extraction(
        job_description_input, llm, clean_text, job_post_prompt, job_post_to_text, parser
    )
    job_description = job_description_doc.page_content

    st.info("Running hybrid retrieval...")

    # Dense Retrieval (Semantic Search)
    retriever = doc_search.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    dense_results = retriever.invoke(job_description)

    # Sparse Retrieval (Keyword Retrievel)
    docs = [doc.page_content for doc in resume_extraction]
    tokenized_docs = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized_docs)

    keywords = job_description.lower().split()
    bm25_scores = bm25.get_scores(keywords)

    top_ids = bm25_scores.argsort()[-3:][::-1]
    sparse_results = [resume_extraction[i] for i in top_ids]

    # Metadata Filtering
    metadata_filtered = [
        c for c in resume_extraction
        if c.metadata["experience"] >= job_description_doc.metadata["experience_required"]
    ]
    
    # Combining Retrieval Results from all, by taking intersection using id
    combined = dense_results + sparse_results + metadata_filtered
    unique_docs = list({doc.metadata["id"]: doc for doc in combined}.values())

    # Candidate Matching using prompt
    candidate_context = "\n\n".join(doc.page_content for doc in unique_docs)

    chain = candidate_matching_prompt | llm

    st.info("Generating final candidate ranking…")

    response = chain.invoke({
        "candidate_context": candidate_context,
        "job_description": job_description
    })

    # Displaying Output to the interface
    st.subheader("Final Ranked Candidates")
    st.write(response.content)
