import numpy as np
from nltk.tokenize import casual_tokenize
import re


# Define the unique tags in the dataset based on the provided entity groups
entity_groups = [
    "ORG",
    "NUM",
    "NAME_EMPLOYEE",
    "LINK",
    "DATE",
    "ACRONYM",
    "MAIL",
    "TELEPHONE",
    "TECH",
    "NAME",
    "PERCENT",
]


def preprocess_text(example, entity_groups):
    words = []
    tags = []
    for i in example["entities"]:
        if len(casual_tokenize(i["word"])) > 1:
            words.append(casual_tokenize(i["word"]))
            tags.append([i["entity_group"]] * len(casual_tokenize(i["word"])))
        else:
            words.append(casual_tokenize(i["word"]))
            tags.append([i["entity_group"]])

    words_flat = [item for sublist in words for item in sublist]
    tags_flat = [item for sublist in tags for item in sublist]
    tags_flat = [entity_groups.index(i) + 1 for i in tags_flat]
    ner_tag_list = []

    for token in casual_tokenize(
        example["text"],
    ):
        if token in words_flat:
            ner_tag_list.append(tags_flat[words_flat.index(token)])
        else:
            ner_tag_list.append(0)

    example["ner_tag"] = ner_tag_list
    example["tokens"] = casual_tokenize(
        example["text"],
    )
    return example


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples["ner_tag"]):
        word_ids = tokenized_inputs.word_ids(
            batch_index=i
        )  # Map tokens to their respective word.
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def prepare_compute_metrics(label_list, seqeval):
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions,
                                  references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    return compute_metrics


def tokens_to_output(tokens, preds, text, entity_groups):
    prediction_output = []
    entities = []
    indexes = [
        index for index, _ in enumerate(preds) if preds[index] != preds[index - 1]
    ]
    final_tokens = [
        tokens[indexes[i]:indexes[i + 1]]
        for i, _ in enumerate(indexes)
        if i != len(indexes) - 1
    ]
    final_preds = [
        preds[indexes[i]:indexes[i + 1]]
        for i, _ in enumerate(indexes)
        if i != len(indexes) - 1
    ]
    for t, p in zip(final_tokens, final_preds):
        if 0 not in p:
            pretok_sent = ""
            for tok in t:
                if tok.startswith("##"):
                    pretok_sent += tok[2:]
                else:
                    pretok_sent += " " + tok
            entities.append(
                {"entity_group": entity_groups[p[0] - 1],
                 "word": pretok_sent[1:]}
            )
    entities = [i for n, i in enumerate(entities) if i not in entities[n + 1:]]
    upd_entities = []
    for i in entities:
        i["word"] = re.sub(r"(\s+)([.,:%/_@\-\+\';?!])(\s+)", r"\2", i["word"])
        i["word"] = re.sub(r"(\s+)([.,:%/_@\-\+\';?!])", r"\2", i["word"])
        i["word"] = (
            i["word"]
            .replace("+ ", "+")
            .replace(" ( ", " (")
            .replace("( ", "(")
            .replace(" ) ", ") ")
            .replace(" )", ")")
        )
        i["start"] = 0
        i["end"] = 0

        if i["entity_group"] in ["TELEPHONE", "NUM", "DATE", "PERCENT"]:
            if any(str.isdigit(c) for c in i["word"]) is False:
                entities.remove(i)
        if i["word"] == ")":
            entities.remove(i)
        if i["word"] == "(":
            entities.remove(i)
        if i["word"] == "":
            entities.remove(i)
    for i in entities:
        for match in re.finditer(re.escape(i["word"]), text.lower()):
            upd_entities.append(
                {
                    "entity_group": i["entity_group"],
                    "word": text[match.start():match.end()],
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    prediction_output.append({"text": text, "entities": upd_entities})
    return prediction_output
