import itertools

import kenlm

# https://huggingface.co/hugohow/5gram-french-kenLM

# model_download = hf_hub_download(
#     repo_id="hugohow/5gram-french-kenLM",
#     filename="lm_5gram_big.bin",
#     local_dir="models",
# )

model_path = "models/lm_5gram_big.bin"
model = kenlm.Model(model_path)


score = [
    ["Je veux", "Je voeux", "Je voudrais"],
    ["Je veux un", "Je veux une", "voudrais un"],
    ["veux un café", "veux une café", "un grand café"],
    ["un café chaud", "un café froid", "un café foi"],
    ["café chaud avec", "café froid avec", "café chaud avai"],
    ["chaud avec du", "froid avec du", "avec de"],
    ["avec du lait", "avec du laie", "avec du les"],
    ["du lait d'amande", "du lait de", "du laie d'amande"],
    ["lait d'amande s'il", "lait d'amande si"],
    ["s'il vous plaît", "s'il vous plait", "si vous plaît"],
]


def all_combinations(list1, list2, list3, phrase):
    return [
        phrase + " " + " ".join(comb) for comb in itertools.product(list1, list2, list3)
    ]


def remove_dupes(combinations):
    t = []
    for i in combinations:
        res = " ".join(list(dict.fromkeys(i.split())))
        t.append(res)
    return t


def give_score(list1, list2, list3, phrase):
    combination = all_combinations(list1, list2, list3, phrase)
    removed_dupes = remove_dupes(combination)
    best_score = -100000
    for i in removed_dupes:
        score = model.score(i)

        if score > best_score:
            best_score = score
            best_phrase = i
    print(f"Best sentence: {best_phrase}, Score: {best_score}")
    return best_phrase


best_phrase = ""
for i in range(len(score) - 2):
    phrase = give_score(score[i], score[i + 1], score[i + 2], best_phrase)
    best_phrase = best_phrase + " " + phrase

best_phrase = " ".join(list(dict.fromkeys(best_phrase.split())))
print(best_phrase)
