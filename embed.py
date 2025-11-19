# taking all the necessary import for generating embedding for the resume
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.functions import load_pdf_file, filter_to_minimal_docs, resume_to_text, resume_features_extraction, embedding_model
from src.prompts import resume_prompt
from src.utils import clean_text
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
# from langchain_community.vectorstores import FAISS

import os

# loading the enviroment variable such gemini key, pinecone key
load_dotenv()

# setting parser for structured output from llm
parser = JsonOutputParser()

# setting up the llm
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",      
    temperature=0.7,
    google_api_key=os.environ["GEMINI_API_KEY"]
) 

# Initializing the resume loading and import feature extraction out of each candidate resume using llm 
# extracted_resume = load_pdf_file(data='resumes')
# filter_resume = filter_to_minimal_docs(extracted_resume)
# resume_extraction = resume_features_extraction(filter_resume,llm,clean_text,resume_prompt,resume_to_text,parser) # extracted resumes document with key features

# print(resume_extraction)

def building_vectordb(resume_extraction,embedding_model):

    # Initializing the embedding model
    embeddings = embedding_model()


    # configuring pinecone to store embedding for each candidate resume
    pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

    # index name for pinecone database
    index_name = "candidate-matching"  

    # creating an index pinecone database if it is present in database it will delete then again create it
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
        pc.create_index(
            name=index_name,
            dimension=3072, # high dimensional embedding to generate or retrieve better context
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        
    # index = pc.Index(index_name)

    # connecting and storing embedding for each candidate resumes
    vectorstore = PineconeVectorStore.from_documents(
        documents=resume_extraction,
        index_name=index_name,
        embedding=embeddings,
    )

    return vectorstore


















