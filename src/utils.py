# preprocessing the resume (resume cleaning)

import re

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]*?>', '', text)
    
    # Remove URLs
    text = re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|'
        r'(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        '',
        text
    )

    # Preserve emails by temporarily replacing them with placeholders
    emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    for i, email in enumerate(emails):
        text = text.replace(email, f"__EMAIL{i}__")

    # Preserve numeric patterns like "2.5 years" or "2-5 years"
    numbers = re.findall(r'\b\d+(?:[\.-]\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?\s*(?:years?|yrs?)\b', text, flags=re.IGNORECASE)
    for i, num in enumerate(numbers):
        text = text.replace(num, f"__NUM{i}__")

    # Remove unwanted special characters (keep . - @ for preserved placeholders)
    text = re.sub(r'[^a-zA-Z0-9@.\-_\s]', ' ', text)

    # Replace multiple spaces with single space
    text = re.sub(r'\s{2,}', ' ', text).strip()

    # Restore numeric patterns
    for i, num in enumerate(numbers):
        text = text.replace(f"__NUM{i}__", num)

    # Restore emails
    for i, email in enumerate(emails):
        text = text.replace(f"__EMAIL{i}__", email)

    return text