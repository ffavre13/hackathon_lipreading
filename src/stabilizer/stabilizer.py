import itertools
import json
from pathlib import Path

import kenlm
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent


# https://huggingface.co/hugohow/5gram-french-kenLM


# model_download = hf_hub_download(
#     repo_id="hugohow/5gram-french-kenLM",
#     filename="lm_5gram_big.bin",
#     local_dir="models",
# )
class Segment(BaseModel):
    start_time: int
    text: str


class Segments(BaseModel):
    segments: list[Segment]


PROMPT = """Tu reçois la transcription d'une lecture labiale en anglais, découpée en segments horodatés (JSON).
Corrige l'orthographe et la grammaire, et ajoute la ponctuation (majuscules, virgules, points).

Règles :
- Renvoie exactement les mêmes segments, dans le même ordre, avec les mêmes start_time.
- Ne fusionne pas et ne découpe pas les segments : modifie uniquement "text".
- Les segments se chevauchent légèrement : si un mot est répété à la fin d'un segment et au début du suivant, supprime le doublon.
- N'invente pas de contenu.

Exemple d'entrée (JSON) :
{
    "segments": [
        {"start_time": 0, "text": "HELLO HOW ARE YOU"},
        {"start_time": 1000, "text": "I AM FINE THANK YOU"},
        {"start_time": 2000, "text": "AND YOU"},
        {"start_time": 3000, "text": "I AM GOOD TOO"}
    ]
}

Exemple de sortie (JSON) :
{
    "segments": [
        {"start_time": 0, "text": "Hello, how are you?"},
        {"start_time": 1000, "text": "I am fine, thank you."},
        {"start_time": 2000, "text": "And you?"},
        {"start_time": 3000, "text": "I am good too."}
    ]
}

Segments :
"""


class SentenceStabilizer:
    def __init__(
        self,
        model_path=str(BASE_DIR / "models" / "lm_5gram_big.bin"),
        max_history=3,
        llm_override=3,
    ):
        self.model = kenlm.Model(model_path)
        self.batch_history = []
        self.stabilized_phrases = []  # [(Start time, Phrase)]
        self.max_history = max_history
        self.llm_last_index = 0
        self.llm_override = llm_override

    def __call__(self, start_time, current_batch):
        return self.get_sentence(start_time, current_batch)

    def get_sentence(self, start_time, current_batch):
        current_batch_with_time = [
            item | {"start_time": start_time} for item in current_batch
        ]

        if len(current_batch) == 0:
            return ""

        if len(self.batch_history) == 0:
            self.batch_history.append(current_batch_with_time)
            self.stabilized_phrases.append((start_time, current_batch[0]["text"]))
            return current_batch[0]["text"]

        self.batch_history.append(current_batch_with_time)
        self.stabilized_phrases.append((start_time, current_batch[0]["text"]))

        self.give_score()

        return self.stabilized_phrases[-1][1]

    def get_phrases_text(self):
        return " ".join([p[1] for p in self.stabilized_phrases])

    def give_score(self):
        best_score = -10000
        best_phrase = []

        for context in itertools.product(*self.batch_history[-self.max_history :]):
            tmp_phrase = ""
            score = 0
            for item in context:
                tmp_phrase += item["text"] + " "
                score += item["score"]

            score += self.model.score(tmp_phrase.strip())

            if score > best_score:
                best_score = score
                best_phrase = context

        self.stabilized_phrases[-self.max_history :] = [
            (item["start_time"], item["text"]) for item in best_phrase
        ]

    def parse_llm_segments(self, content):
        content = content[content.find("[") : content.rfind("]") + 1]
        return [
            {"start_time": item["start_time"], "text": item["text"]}
            for item in json.loads(content)
        ]

    def llm_stabilize(self, client, last=False):
        limit = 0 if last else self.max_history

        llm_new_index = len(self.stabilized_phrases) - limit

        if len(self.stabilized_phrases) - limit <= 0:
            return

        stabilized_phrases_to_send = self.stabilized_phrases[
            self.llm_last_index : llm_new_index
        ]

        if len(stabilized_phrases_to_send) == 0:
            return

        stabilized_phrases_to_send = [
            {"start_time": start_time, "text": text}
            for start_time, text in stabilized_phrases_to_send
        ]

        try:
            response = client.chat(
                model="gemma4",
                messages=[
                    {
                        "role": "user",
                        "content": PROMPT
                        + json.dumps(stabilized_phrases_to_send, ensure_ascii=False),
                    }
                ],
                format=Segments.model_json_schema(),
                options={"temperature": 0},
            )
            corrected = self.parse_llm_segments(response.message.content)
        except Exception as e:
            print(f"[LLM] erreur : {e}")
            return

        for i, j in enumerate(range(self.llm_last_index, llm_new_index)):
            self.stabilized_phrases[j] = (
                corrected[i]["start_time"],
                corrected[i]["text"],
            )

        self.llm_last_index = max(0, llm_new_index - self.llm_override)
