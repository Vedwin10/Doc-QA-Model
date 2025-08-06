import re
import unicodedata
from typing import Optional, List
from contractions_dict import contractions
from symspellpy import SymSpell, Verbosity

class QueryPreprocessor:

  def __init__(self,
               min_query_length: int = 2,
               max_query_length: int = 256,
               enable_spell_check: bool = False):
    self.min_query_length = min_query_length
    self.max_query_length = max_query_length

    # use symspellpy to load in spelling dictionary for spellcheck
    if enable_spell_check:
      try:
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self.sym_spell.load_dictionary('frequency_dictionary_en_82_765.txt', term_index=0, count_index=1)
        self.spell_check_enabled = True
      except ImportError:
        print("Spell Check disabled. Install symspellpy for spell checking.")
        self.spell_check_enabled = False
    else:
      self.spell_check_enabled = False

  # main function
  def preprocess(self, query) -> List[str]:
      if not isinstance(query, str):
        raise TypeError(f"Query must be string, got {type(query)}")
      
      # Pipeline of processing steps
      query = self.normalize_encoding(query)
      query = self.expand_contractions(query)
      query = self.clean_special_chars(query)
      query = self.normalize_whitespace(query)
      questions = self.split_query(query)
      corrected_questions = []
      for question in questions:
        cq = self.correct_spelling(question)
        cq = self.validate_query(cq)
        corrected_questions.append(cq)

      return corrected_questions

  # use unicodedata to convert query to NFC
  def normalize_encoding(self, query):
    query = query.replace("’", "'")
    return unicodedata.normalize('NFC', query)

  # expand contractions for smoother processing
  def expand_contractions(self, query):
    self.contractions = contractions
    pattern = re.compile('(%s)' % '|'.join(map(re.escape, self.contractions.keys())), flags=re.IGNORECASE)

    def replace(match):
      match_text = match.group(0).lower()
      return self.contractions.get(match_text, match_text)

    return pattern.sub(replace, query)

  # use re to get rid of problematic special chars
  def clean_special_chars(self, query):
    return re.sub(r'[^\w\s?!\.,\'\"\(\)\[\]\{\}\-]', '', query)    # keeps letters, numbers, ?, !, ., ,, ', ", (, ), [, ], {, }

  # use re to get rid of extra whitespace
  def normalize_whitespace(self, query):
    query = re.sub(r'\s+', ' ', query)
    return query.strip()

  # use symspellpy to correct spelling
  def correct_spelling(self, query):
    corrected_terms = []
    for term in query.lower().split():
      if len(term) > 3:
        suggestions = self.sym_spell.lookup(term, Verbosity.TOP, max_edit_distance=2)
        corrected_terms.append(suggestions[0].term if suggestions else term)
      else:
        corrected_terms.append(term)
            
    return ' '.join(corrected_terms)

  # split into array of questions    
  def split_query(self, query: str) -> List[str]:
    questions = re.findall(r'[^?]+?\?', query)
    return [q.strip() for q in questions if q.strip()]

  # make sure query is within length bounds and contains question marks for questions
  def validate_query(self, query):
      if len(query) < self.min_query_length:
        return None
      if len(query) > self.max_query_length:
        query = query[:self.max_query_length].rsplit(' ', 1)[0] + '...'
          
      if any(q_word in query.lower().split() for q_word in ['who', 'what', 'where', 'when', 'why', 'how']):
        if not query.strip().endswith('?'):
          query += '?'
      
      return query