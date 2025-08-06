import json
import random
from typing import List, Dict, Tuple, Literal, Optional

class SquadParser:
    def __init__(self, json_file: str):
        with open(json_file, "r", encoding="utf-8") as f:
            squad_data = json.load(f)
        self.data = squad_data.get("data", [])
        self.context_question_pairs = self._extract_context_question_pairs()

    def _extract_context_question_pairs(self) -> List[Tuple[str, List[Dict]]]:
        pairs = []
        for article in self.data:
            for paragraph in article.get("paragraphs", []):
                context = paragraph.get("context", "")
                qas = paragraph.get("qas", [])
                pairs.append((context, qas))
        return pairs

    def get_random_pairs(
        self, 
        n: int = 10, 
        filter_by: Optional[Literal["answerable", "unanswerable", "both"]] = "both"
    ) -> List[Tuple[str, List[Dict]]]:
        filtered_pairs = []

        for context, qas in self.context_question_pairs:
            if filter_by == "answerable":
                filtered_qas = [q for q in qas if not q.get("is_impossible", False)]
            elif filter_by == "unanswerable":
                filtered_qas = [q for q in qas if q.get("is_impossible", False)]
            else:  # both
                filtered_qas = qas

            if filtered_qas:
                filtered_pairs.append((context, filtered_qas))

        return random.sample(filtered_pairs, min(n, len(filtered_pairs)))


# # Load random question sets from SQuAD
# from squad_parser import SquadParser

# parser = SquadParser("train-v2.0.json")
# random_pairs = parser.get_random_pairs(n=5)    # pass in "answerable", "unanswerable", or "both"

# # retriever topk = 30
# # final topk = 5

# for context, questions in random_pairs:
#     for q in questions:
#         query = q