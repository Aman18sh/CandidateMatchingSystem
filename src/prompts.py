from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

resume_prompt = PromptTemplate(
    template=
    """
    ### SCRAPED TEXT FROM RESUME:
    {resume_data}
    ### INSTRUCTION:
    The scraped text above is from a candidate's resume.
    Your task is to extract the key information and return it in a structured JSON format with the following fields:

    - `name`: Full name of the candidate.
    - `email`: Candidate’s email address.
    - `phone`: Candidate’s phone number (if available).
    - `role`: Current or most recent job titles / designation with company details.
    - `experience_years`: mention experience years in integer value if avaible else mention 0.
    - `skills`: List of technical and soft skills mentioned (normalized and deduplicated).
    - `education`: Highest qualification and institution.
    - `projects`: List of key projects with their titles and short descriptions.
    - `certifications`: List of certifications or courses (if any).
    - `summary`: A brief 2–3 sentence professional summary based on the resume.

    if any of the field is missing then write "not available" as a string

    
    {format_instructions}
    """,
    input_variables=["resume_data"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


job_post_prompt = PromptTemplate(
    template="""
    ### SCRAPED TEXT FROM WEBSITE:
    {job_post}

    ### INSTRUCTION:
    The scraped text above is from the careers or jobs page of a company website.
    Your task is to extract job postings and return them in **strict JSON format**.

    job posting should be represented with the following keys, 
    job_posting is the main key and its value is the list containing all the following field:
    
    - `role`: The job title or designation.
    - `company`: Name of the company (if mentioned).
    - `location`: Job location (city, state, or remote/hybrid if specified).
    - `experience_required`: Required experience in years as integer. if there is no experienced mention 0
    - `skills`: A list of technical and soft skills explicitly mentioned.
    - `skill_classification`: Categorize the extracted skills into the following buckets:
        - `must_have`: essential or required skills.
        - `important`: valuable but not strictly required skills.
        - `nice_to_have`: additional or preferred skills.
        if any of skill not availble mention ["None"]
    - `description`: A clean summary (2–3 sentences) combining key responsibilities and qualifications.
    - `employment_type`: Full-time, Part-time, Internship, Contract, etc., if available.
    - `posted_date`: Date of posting (if available).


    ### REQUIRED JSON FORMAT:
    {format_instructions}

    ### RETURN ONLY VALID JSON, NO PREAMBLE:
    """,
    input_variables=["job_post"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


    # If multiple job roles are mentioned, return an array of job posting objects.

candidate_matching_prompt = PromptTemplate(
    template="""
    Evaluate and score the candidates through a structured, multi-dimensional framework 
    that measures alignment with the job requirements.

    Use the following criteria:
    - Skill alignment
    - Experience relevance
    - Educational fit
    - Role compatibility
    - Overall suitability

    For each candidate:
    - Summarize strengths
    - Identify skill gaps
    - Provide an overall fit score in percentage out of 100%
    - Provide confidence level in the score

    After evaluating all candidates:
    - Produce a ranked list from strongest to weakest match
    - Include detailed explanations for ranking decisions

    Job Description:
    {job_description}

    Candidates:
    {candidate_context}
    """,
    input_variables=['job_description','candidate_context']

)