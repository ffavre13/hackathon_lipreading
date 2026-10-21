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

score = [
    [("Je veux", -3.42), ("Je voeux", -6.81), ("Je voudrais", -3.75)],
    [("Je veux un", -4.15), ("Je veux une", -4.89), ("voudrais un", -5.12)],
    [
        ("veux un café", -5.12),
        ("veux une café", -8.45),
        ("un grand café", -5.88),
    ],
    [
        ("un café chaud", -5.21),
        ("un café froid", -5.48),
        ("un café foi", -9.10),
    ],
    [
        ("café chaud avec", -5.85),
        ("café froid avec", -6.12),
        ("café chaud avai", -10.25),
    ],
    [
        ("chaud avec du", -5.30),
        ("froid avec du", -5.60),
        ("avec deux", -7.42),
    ],
    [
        ("avec du lait", -4.18),
        ("avec du laie", -8.95),
        ("avec du les", -9.12),
    ],
    [
        ("du lait d'amande", -5.02),
        ("du lait de", -4.75),
        ("du laie d'amande", -9.80),
    ],
    [("lait d'amande s'il", -5.40), ("lait d'amande si", -5.85)],
    [
        ("s'il vous plaît", -3.12),
        ("s'il vous plait", -3.98),
        ("si vous plaît", -4.50),
    ],
]


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

    for (f1, score_f1), (f2, score_f2), (f3, score_f3) in itertools.product(
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


best_phrase = ""
for i in range(len(score) - 2):
    phrase = give_score(score[i], score[i + 1], score[i + 2], best_phrase)
    best_phrase = remove_dupe(best_phrase, phrase)
    best_phrase = sentence_limit(best_phrase)
print(best_phrase)
