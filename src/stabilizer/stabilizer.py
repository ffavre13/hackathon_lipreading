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

model_path = BASE_DIR / "models" / "lm_5gram_big.bin"
model = kenlm.Model(str(model_path))

score = [()]


def get_sentence(current_batch, current_phrase, batch_history, max_history=3):
    if len(batch_history) == 0:
        batch_history.append(current_batch)
        return current_batch[0][0], batch_history

    if len(batch_history) == 1:
        phrase = batch2_score(batch_history[0], current_batch, current_phrase)
        batch_history.append(current_batch)
        return phrase, batch_history

    if current_batch[0][2] == True:
        phrase = give_score(
            batch_history[-2], batch_history[-1], current_batch, current_phrase
        )
        temp_phrase = remove_dupe(current_phrase, phrase)
        batch_history.append(current_batch)

        last = last_batch(batch_history[-2:], temp_phrase)
        final_phrase = remove_dupe(phrase, last)

        return final_phrase, batch_history

    if len(batch_history) >= max_history:
        batch_history.pop(0)

    batch_list = batch_history[-2:]
    phrase = give_score(batch_list[0], batch_list[1], current_batch, current_phrase)
    batch_history.append(current_batch)
    return phrase, batch_history


def last_batch(batch_history, current_phrase):
    best_score = -10000
    best_word = ""

    if len(batch_history) == 2:
        for (f1, s1, _), (f2, s2, _) in itertools.product(
            batch_history[0], batch_history[1]
        ):
            temp_phrase = remove_dupe(current_phrase, f1)
            temp_phrase = remove_dupe(temp_phrase, f2)
            total_score = s1 + s2 + model.score(temp_phrase)
            if total_score > best_score:
                best_score = total_score
                best_word = remove_dupe(f1, f2)

    elif len(batch_history) == 1:
        for f1, s1, _ in batch_history[0]:
            temp_phrase = remove_dupe(current_phrase, f1)
            total_score = s1 + model.score(temp_phrase)
            if total_score > best_score:
                best_score = total_score
                best_word = f1

    return best_word


def batch2_score(list1, list2, current_phrase):
    best_score = -10000
    best_phrase = ""
    word_f1 = list1[0][0]

    for (f1, score_f1, _), (f2, score_f2, _) in itertools.product(list1, list2):
        temp_phrase = remove_dupe(current_phrase, f1)
        temp_phrase = remove_dupe(temp_phrase, f2)

        score = model.score(temp_phrase)

        total_score = score_f1 + score_f2 + score

        if total_score > best_score:
            best_score = total_score
            best_phrase = temp_phrase
            word_f1 = f1

    # print(f"Best sentence: {best_phrase}, Score: {best_score}")
    return word_f1


def remove_dupe(seq1, seq2):
    w1, w2 = seq1.split(), seq2.split()

    max_overlap = min(len(w1), len(w2))
    for i in range(max_overlap, 0, -1):
        if w1[-i:] == w2[:i]:
            return " ".join(w1 + w2[i:])

    return " ".join(w1 + w2)


def give_score(list1, list2, list3, current_phrase):
    best_score = -10000
    best_phrase = ""
    word_f1 = list1[0][0]

    for (f1, score_f1, _), (f2, score_f2, _), (f3, score_f3, _) in itertools.product(
        list1, list2, list3
    ):
        temp_phrase = remove_dupe(current_phrase, f1)
        temp_phrase = remove_dupe(temp_phrase, f2)
        temp_phrase = remove_dupe(temp_phrase, f3)

        score = model.score(temp_phrase)

        total_score = score_f1 + score_f2 + score_f3 + score

        if total_score > best_score:
            best_score = total_score
            best_phrase = temp_phrase
            word_f1 = f1

    # print(f"Best sentence: {best_phrase}, Score: {best_score}")
    return word_f1


def sentence_limit(sentence, limit=100):
    words = sentence.split()
    diff = len(words) - limit
    if diff > 0:
        sentence = " ".join(words[diff:])
    return sentence


def fill_batch():
    list_batch = score


best_phrase = ""
batch_history = []

for batch in score:
    phrase, batch_history = get_sentence(batch, best_phrase, batch_history)
    best_phrase = remove_dupe(best_phrase, phrase)

    display_phrase = remove_dupe(best_phrase, batch[0][0])

print(display_phrase)

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
