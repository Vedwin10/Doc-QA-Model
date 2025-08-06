from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict

class TextChunker:

  def __init__(self,
               chunk_size: int = 512,
               chunk_overlap: int = 128,
               separators: List[str] = None):
    
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

    self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    self.splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=self.separators,
        length_function=len,
        is_separator_regex=False
    )
    # TODO

  # function to create chunks with metadata
  def chunk_text(self, text: str, metadata: dict) -> List[Dict]:
    if not text:
      return []
    
    documents = self.splitter.create_documents([text], [metadata])

    chunks = []
    for i, doc in enumerate(documents):
      chunk_metadata = doc.metadata.copy()
      chunk_metadata["chunk_index"] = i
      chunks.append({
          "text": doc.page_content,
          "metadata": chunk_metadata
      })
    
    return chunks