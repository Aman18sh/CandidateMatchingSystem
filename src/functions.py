import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List
from langchain_core.documents import Document


# loading all the resumes
def load_pdf_file(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    return documents


# filter out the relevant metadata from the loaded resumes
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' and 'total_pages' in metadata and the original page_content
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        t_pages = doc.metadata.get("total_pages")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src,"total_pages":t_pages}
            )
        )
    return minimal_docs


# function to convert resume_json to regular text
def resume_to_text(resume_json):
    return f"""
    Name: {resume_json.get('name')}
    Role: {resume_json.get('role')}
    Experience: {resume_json.get('experience_years')} years
    Skills: {", ".join(resume_json.get('skills', []))}
    Education: {resume_json.get('education')}
    Projects: {[p['title'] for p in resume_json.get('projects', [])]}
    Summary: {resume_json.get('summary')}
    Certifications: {", ".join(resume_json.get('certifications', []))}
    """  

# extracting key features from each resumes (resume -> Json -> text)

def resume_features_extraction(docs,llm_model,resume_cleaner,prompt_template,resume_to_text,parser):
    all_candidates = []
    count=1
    for doc in docs: # each doc represents a resume document, docs is collection of resume documents in a list
        clean_resume = resume_cleaner(doc.page_content)  # clean the resume in nice format (funtion defined in util.py)
        chain = prompt_template | llm_model | parser     # langchain chain expression
        resume_json = chain.invoke({'resume_data':clean_resume}) # give the keys features for each candidate in json form 
        resume_text = resume_to_text(resume_json) # convert json format into regular text with function json to text
        document_resume = Document(metadata={"id":count,"name":resume_json['name'],"experience":resume_json['experience_years'],"skills":resume_json['skills']},page_content=resume_text)   # convert into each resume into document form for embedding
        all_candidates.append(document_resume)      # append each document resume in list
        count+=1
    return all_candidates


# function to convert post_json to regular text
def job_post_to_text(job_json):
    return f"""
Role: {job_json.get('role')}
Company: {job_json.get('company')}
Location: {job_json.get('location')}
Experience Required: {job_json.get('experience_required')}
Employment Type: {job_json.get('employment_type')}
Posted Date: {job_json.get('posted_date')}

Skills: {", ".join(job_json.get('skills', []))}

Skill Classification:
  Must Have: {", ".join(job_json.get("skill_classification", {}).get("must_have", []))}
  Important: {", ".join(job_json.get("skill_classification", {}).get("important", []))}
  Nice To Have: {", ".join(job_json.get("skill_classification", {}).get("nice_to_have", []))}

Description:
{job_json.get('description')}
""".strip()
  

# extracting key features from job post

def jobpost_feature_extraction(doc,llm_model,post_cleaner,prompt_template, job_post_to_text, parser):
    clean_post = post_cleaner(doc)
    chain = prompt_template | llm_model | parser
    post_json = chain.invoke({'job_post':clean_post})
    print(post_json)
    post_text = job_post_to_text(post_json['job_posting'][0])
    document_post = Document(metadata={"experience_required":post_json['job_posting'][0]['experience_required'],"employment_type":post_json['job_posting'][0]['employment_type'],"posted_date":post_json['job_posting'][0]["posted_date"]},page_content=post_text)
    print(type(post_json['job_posting'][0]['experience_required']))
    return document_post

# setting up the embedding model

def embedding_model():
    embeddings =  GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT",
        google_api_key=os.environ["GEMINI_API_KEY"]
    )

    return embeddings



