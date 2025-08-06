from typing import List, Dict

class PromptEngineer:
  def __init__(self, 
               system_prompt: str = None,
               max_content_length: int = 160000,    #164K tokens for input deepseek-r1, use 160K
               include_relevance: bool = True):
    self.system_prompt = system_prompt or self.default_system_prompt()
    self.max_context_length = max_content_length
    self.include_relevance = include_relevance

  # default prompt for model role
  @staticmethod
  def default_system_prompt() -> str:
    return """You are an expert research assistant. Your task is to answer questions based ONLY on the provided context.
RULES:
1. Answer the question using ONLY the context provided
2. If the question cannot be answered with the context, say "I could not find an answer in the provided document(s)"
3. Be concise but comprehensive
4. Never invent information not present in the context"""

  # function to format the retrieved context chunks
  def format_context(self, context_chunks: List[Dict]) -> str:
    context_str = ""
    for i, chunk in enumerate(context_chunks):
      text = self.truncate_text(chunk['chunk'], self.max_context_length)

      metadata = chunk['metadata']

      relevance_info = ""
      if self.include_relevance and 'relevance' in chunk:
        relevance_info = f" [Relevance: {chunk['relevance']:.2f}]"

      context_str += (
          f"### CONTEXT {i+1}{relevance_info}\n"
          f"CONTENT: {text}\n\n"
      )
    return context_str.strip()

  # if text exceeds token limit, truncate it
  @staticmethod
  def truncate_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
      return text

    last_period = text.rfind('.', 0, max_length)
    last_question = text.rfind('?', 0, max_length)
    last_exclaim = text.rfind('!', 0, max_length)
    sentence_end = max(last_period, last_question, last_exclaim)

    if sentence_end > 0:
      return text[:sentence_end+1] + " [TRUNCATED]"
    return text[:max_length] + " [TRUNCATED]"

  # function to format prompt with queries
  def format_prompt(self, processed_query: str, context_chunks: List[Dict]) -> str:
    context_str = self.format_context(context_chunks)

    return f"""
{self.system_prompt}

CONTEXT DOCUMENTS:
{context_str}

QUESTION: {processed_query}

ANSWER:
""".strip()