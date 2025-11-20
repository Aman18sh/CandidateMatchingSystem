**AI Driven Candidate Matching System**


An Intelligent Candidate Matching System that evaluates and ranks candidates based on job requirements using the power of large language model (LLMs: gemini-2.5-flash), hybrid search and vector embeddings.

**-------------------------------------------------------------------------------------------------------------------------------------**

**Overview**

This system automates candidate screening by:

- Extracting structured information from resumes
- Understanding job description
- Converting both into meaningful text representations
- Embedding resumes into Pinecone for semantic retrievel
- Using hybrid search (semantic + keyword + metadata) to retrieve relevant candidates (intersection of all search)
- Applying LLM-based reasoning to rank and evaluate candidates
- Generating explainable reports for each candidate

**Architecture**

The entire workflow consists of three major pipelines:

1. Resumes Ingestion Pipeline
2. Job Post Processing Pipeline
3. Candidate Matching & Ranking Pipeline

workflow: https://www.figma.com/board/Cilwjyp2GNeSzfSSbsL7Ez/Candidate-Matching-Workflow?node-id=0-1&t=m6FZHMH2ZCbcHVb2-1

**Job Processing Pipeline**

When users upload resumes (PDF files), each resumes goes through the following steps
1. PDF Text Extraction
using langchain ByPDFLoader to extract raw text from the PDF

2. Text Processing
- Clean and Normalize the text
- Removes Symbols
- Fixes Spacing
- Prepares input for LLMs

3. Features Extraction through LLM
The LLM is prompted to generate structured JSON containing:
- Name
- Skill
- Education
- Work Experience
- Projects
- Domain/Specialization
- Summary

4. JSON to Descriptive Text Conversion
- Structured data is converted into plain natural language text.
- This will ensures that models understand the information semantically.

5. Document Creation
Each resumes becomes a langchain document containing
- page_content: candidate description
- metadata: candidate id, name, experience, domain, skills

6. Vector Embedding and Storage
- Embed the candidate document using an embedding model
- Stores vector + metadata in pinecone (each vector correspond to each candidate profile and metadata is stored along side for metadata filtering)

**Job Post Processing Pipeline**

when the user enter the job description:

1. Text Cleaning
- Removes noise and formats job post consistently.

2. Feature Extraction via LLM
LLM extracts structured JSON containing.
- Required skills (musthave, important, nice to have)
- Required experience
- Responsibilities
- Domain
- Preferred Qualifications

3. JSON to Natural Text Conversion
converts structured job description into readable descriptive text.

4. Job Document Creation
a langchain document is created with:
- page_content: job description
- metadata: experience_required, job_role, skills


**Candidate Matching and Ranking**

Once all candidate vectors are stored and a job post is processedm, hybrid candidate search will be performed:

***Hybrid Retrievel***

The system retrieves candidate using a hybrid search consisting of:
1. Dense Semantic Search (vector search)
Finds resumes semantically similar to the job requirements.

2. Sparse Keyword Search (BM25)
Matches exact keywords from job post and resume.

3. Metadata Filtering
- Filtered by minimum no of experience years
- domain alignment


4. Combined Candidate Pooling

Results from all 3 searches are merged.(Intersection)
Duplicates removed using candidate metadata (id).

This becomes the context for the LLM reasoning.


***LLM based Candidate Ranking***

The combined candidate profiles + job description are passed into the LLM with a *candidate_matching_prompt*

*The LLM Generates:*

- Ranked list of candidates
- Fit scores
- Strengths
- Skill gaps
- Alignment with a job role
- confidence score
- over all recommendation

This produces an explainable and transparent evaluation.



**UI Layer Integration using Streamlit**

The UI Provides:

- Resume upload functionality
- Job Description input field
- Status updates during pipeline execution
- Final ranked list of candidates with detailed insights

Candidates are displayed with:

- Match score
- Key strengths
- Skill gaps
- Experience Summary
- Explaination of why they fit or don't fit


**Technology Stack**

- LLM: gemini-2.5-flash
- Text Extraction: Langchain PyPDFLoader
- Preprocessing: Custom Cleaning Pipeline
- Embedding: Embedding model(gemini-embedding-001, 3072 Dimension vector) 
- Vector DB: Pinecone
- Keyword search: BM25
- Orchestration: Langchain
- UI: Streamlit









**Thank you**

*Best Regards* |
*Aman Sharma* |
*Digital University Kerala (formerly IIITMK)* |
*aman.ds24@duk.ac.in, amansharmaaa9313@gmail.com* |
*+91 8799706360*

- ***Linkedin:*** *https://www.linkedin.com/in/aman-sharma-4272ab231*
- ***Portfolio:*** *https://aman18sh.github.io/portfolio/*
- ***Medium:*** *https://medium.com/@amansharmaaa9313*
- ***Github:*** *https://github.com/Aman18sh*
- ***BI Portfolio:*** *https://my.novypro.com/amansharma-1*

