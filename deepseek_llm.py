import requests
import time
from typing import Dict, Any

class DeepSeekLLM:

  def __init__(self,
               api_key: str,
               model: str = "deepseek/deepseek-r1-0528:free",       # deepseek-reasoner for deepseek direct
               base_url: str = "https://openrouter.ai/api/v1",      # https://api.deepseek.com for deepseek direct
               max_retries: int = 3,
               timeout: int = 120):
    self.api_key = api_key
    self.model = model
    self.base_url = base_url
    self.max_retries = max_retries
    self.timeout = timeout
    self.headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

  # function to generate a response
  def generate(self,
               prompt: str,
               max_tokens: int = 160000,    #164K token output for deepseek-r1
               temperature: float = 0.7,
               top_p: float = 0.9,
               stop: list = None) -> Dict[str, Any]:
    payload = {
      "model": self.model,
      "messages": [{"role": "user", "content": prompt}],
      "max_tokens": max_tokens,
      "temperature": temperature,
      "top_p": top_p,
      "stop": stop or [],
      "stream": False
    }

    for attempt in range(self.max_retries):
      try:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
      except requests.exceptions.RequestException as e:
        if attempt < self.max_retries - 1:
          wait_time = 2 ** attempt
          print(f"Error: {str(e)}. Retrying in {wait_time} seconds...")
          time.sleep(wait_time)
        else:
          raise RuntimeError(f"API request failed after {self.max_retries} attempts: {str(e)}")

  
  # function to extract answer from API response dict
  def extract_answer(self, response: Dict) -> str:
    try:
      return response['choices'][0]['message']['content'].strip()
    except (KeyError, IndexError, TypeError):
      print("Warning: Unexpected response format. Returning raw content.")
      return str(response)

  # function to format extracted answer
  def format_answer(self, raw_answer: str) -> str:
    last_period = raw_answer.rfind('.')
    last_question = raw_answer.rfind('?')
    last_exclaim = raw_answer.rfind('!')
    sentence_end = max(last_period, last_question, last_exclaim)

    if sentence_end > 0:
      clean_answer = raw_answer[:sentence_end+1]
    else:
      clean_answer = raw_answer

    import re
    clean_answer = re.sub(r'\[[^\]]+\]', '', clean_answer)  # Remove [citation] formats
    clean_answer = re.sub(r'\([^)]+\)', '', clean_answer)   # Remove (citation) formats
    clean_answer = clean_answer.replace("<|im_end|>", "").replace("<|im_start|>", "")
    return clean_answer.strip()

  # main function to show prompt, tokens, and answer
  def answer_query(self, prompt: str, max_tokens: int = 160000) -> str:
    response = self.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.6,    # 0.6 recommended for deepseek-r1 model
        top_p=0.9,
        stop=["<|im_end|>", "###", "SOURCES:"]
    )

    raw_answer = self.extract_answer(response)
    return self.format_answer(raw_answer)