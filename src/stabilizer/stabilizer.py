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
Les segments sont des morceaux consécutifs d'un même texte continu : une phrase peut commencer dans un segment et se terminer dans un autre.

Ta tâche :
1. Supprimer les doublons dus au chevauchement (PRIORITAIRE).
2. Corriger l'orthographe et la grammaire.
3. Ajouter la ponctuation et les majuscules, en pensant au texte complet obtenu en mettant les segments bout à bout.
4. Corriger les erreurs de transcription si nécessaire, mais uniquement si tu es sûr à 100% de la correction. Ne change pas le sens du texte.

Méthode pour les doublons :
Les segments se chevauchent : la fin d'un segment est souvent répétée au début du suivant.
Pour chaque paire de segments consécutifs, compare les 1 à 4 derniers mots du premier avec les 1 à 4 premiers mots du second.
S'ils sont identiques, supprime-les du début du second segment. Ça peut concerner plusieurs mots d'un coup (par exemple "GO TO THE").
Si un segment ne contient plus rien après suppression, renvoie-le avec "text": "".
Supprimer ces doublons n'est pas modifier le contenu : c'est obligatoire.

Règles :
- Renvoie tous les segments, dans le même ordre, avec les mêmes start_time.
- Ne fusionne pas et ne découpe pas les segments : modifie uniquement "text".
- Réponds uniquement avec du JSON, au même format que l'exemple de sortie.

Exemple d'entrée :
{
    "segments": [
        {"start_time": 0, "text": "I THINK WE SHOULD GO TO THE"},
        {"start_time": 1000, "text": "TO THE BEACH TOMORROW MORNING"},
        {"start_time": 2000, "text": "MORNING IF THE WEATHER IS"},
        {"start_time": 3000, "text": "WEATHER IS"},
        {"start_time": 4000, "text": "NICE WHAT DO YOU THINK"}
    ]
}

Exemple de sortie :
{
    "segments": [
        {"start_time": 0, "text": "I think we should go to the"},
        {"start_time": 1000, "text": "beach tomorrow morning"},
        {"start_time": 2000, "text": "if the weather is"},
        {"start_time": 3000, "text": ""},
        {"start_time": 4000, "text": "nice. What do you think?"}
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
