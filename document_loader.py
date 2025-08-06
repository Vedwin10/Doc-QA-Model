import os
import fitz
import re
import unicodedata

class DocumentLoader:

  # main function
  def load_document(self, file_path):
    if not os.path.exists(file_path):
      raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.lower().endswith('.pdf'):
      raise ValueError("Only PDF files are supported in this model")

    try:
      doc = fitz.open(file_path)
      full_text = ""
      metadata = {
          "page_count": len(doc),
          "author": doc.metadata.get("author", ""),
          "title": doc.metadata.get("title", "")
      }

      for page in doc:
        full_text += page.get_text() + "\n\n"

      return self.clean_text(full_text), metadata

    except Exception as e:
      if "password" in str(e).lower():
        raise RuntimeError("Password-protected PDFs not supported")
      elif "invalid format" in str(e).lower():
        raise RuntimeError("File is not a valid PDF")
      else:
        raise RuntimeError(f"PDF \'{file_path}\' processing failed: {str(e)}")

  # function to clean up pdf-extracted text
  def clean_text(self, text):
    # normalize to NFC to maintain consistency with query normalization
    text = unicodedata.normalize('NFC', text)
    # remove number-only lines
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    # fix hyphenated words across lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # remove extra whitespace
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()