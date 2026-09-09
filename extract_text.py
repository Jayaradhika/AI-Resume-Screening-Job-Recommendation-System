"""
Text Extraction Module
Extracts raw text from uploaded resume files (PDF or DOCX).
"""

import os
import PyPDF2
import docx


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_docx(file_path: str) -> str:
    document = docx.Document(file_path)
    return "\n".join(p.text for p in document.paragraphs)


def extract_text(file_path: str) -> str:
    """Dispatch extraction based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or DOCX.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(extract_text(sys.argv[1]))
