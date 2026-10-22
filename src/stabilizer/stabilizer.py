import itertools
from pathlib import Path

import kenlm

BASE_DIR = Path(__file__).resolve().parent

# https://huggingface.co/hugohow/5gram-french-kenLM

# model_download = hf_hub_download(
#     repo_id="hugohow/5gram-french-kenLM",
#     filename="lm_5gram_big.bin",
#     local_dir="models",
# )

class SentenceStabilizer:
    def __init__(self, model_path=str(BASE_DIR / "models" / "lm_5gram_big.bin"), max_history=3):
        self.model = kenlm.Model(model_path)
        self.batch_history = []
        self.stabilized_phrases = [] # [(Start time, Phrase)]
        self.max_history = max_history

    def __call__(self, start_time, current_batch):
        return self.get_sentence(start_time, current_batch)

    def get_sentence(self, start_time,current_batch):
        current_batch_with_time = [item | {"start_time": start_time} for item in current_batch]

        if len(current_batch) == 0:
            self.stabilized_phrases.append((start_time, ""))
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
                
        for context in itertools.product(*self.batch_history[-self.max_history:]):
            tmp_phrase = ""
            score = 0
            for item in context:
                tmp_phrase += item["text"] + " "
                score += item["score"]
                
            score += self.model.score(tmp_phrase.strip())
            
            if score > best_score:
                best_score = score
                best_phrase = context
                    
        self.stabilized_phrases[-self.max_history:] = [(item["start_time"], item["text"]) for item in best_phrase]

# api_key = os.getenv("OLLAMA_API")

# client = Client(
#     host="http://localhost:11434",
#     headers={"Authorization": f"Bearer {api_key}"},
# )

# stream = client.chat(
#     model="gemma4:cloud",
#     messages=[
#         {
#             "role": "user",
#             "content": "Corrige la phrase suivante si nécessaire et renvoie seulement la phrase corrigée : "
#             + best_phrase,
#         },
#     ],
#     stream=True,
# )

# for chunk in stream:
#     print(chunk["message"]["content"], end="")
