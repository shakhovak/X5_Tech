import random
import re
import json


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

# task is split inot ATE - identification of named entities
# ASC - classification of the identified entity category
# ABSA - both idebtification and classification


def createATE_dataset(sample, prompts_file_path):
    """function to create ATE dataset"""
    with open(prompts_file_path, encoding="UTF-8") as fp:
        template = json.load(fp)

    num = random.randint(1, len(template))
    instruction = template["ATE"][str(num)]

    sample["aspect_list"] = ",".join([item["word"] for item in sample["entities"]])
    sample["ner_list"] = ",".join([item["entity_group"] for item in sample["entities"]])
    sample["aspect_ner_list"] = ",".join(
        [f"{item['word']}:{item['entity_group']}" for item in sample["entities"]]
    )
    sample["aspect_ner_output"] = f"Ответ: \n{sample['aspect_list']}</s>"
    sample["aspect_ner_input"] = (
        f"<LM>Задача: Извлечение именованных сущностей \n{instruction}\n{sample['text']}\n"
    )
    return sample


def createASC_dataset(sample, prompts_file_path):
    """function to create ASC dataset"""
    with open(prompts_file_path, encoding="UTF-8") as fp:
        template = json.load(fp)

    num = random.randint(1, len(template))
    instruction = template["ASC"][str(num)]

    sample["aspect_list"] = ",".join([item["word"] for item in sample["entities"]])
    sample["ner_list"] = ",".join([item["entity_group"] for item in sample["entities"]])
    sample["aspect_ner_list"] = ",".join(
        [f"{item['word']}:{item['entity_group']}" for item in sample["entities"]]
    )
    sample["aspect_ner_output"] = f"Ответ: \n{sample['ner_list']}</s>"
    sample["aspect_ner_input"] = (
        f"<LM>Задача: Классификация именованных сущностей \n{instruction}\nТекст: \n{sample['text']}\nВыделенные именованные сущности: {sample['ner_list']}\n"
    )
    return sample


def createABSA_dataset(sample, prompts_file_path):
    """function to create ABSA dataset """
    with open(prompts_file_path, encoding="UTF-8") as fp:
        template = json.load(fp)

    num = random.randint(1, len(template))
    instruction = template["ABSA"][str(num)]

    sample["aspect_list"] = ",".join([item["word"] for item in sample["entities"]])
    sample["ner_list"] = ",".join([item["entity_group"] for item in sample["entities"]])
    sample["aspect_ner_list"] = ",".join(
        [f"{item['word']}:{item['entity_group']}" for item in sample["entities"]]
    )

    sample["aspect_ner_output"] = f"Ответ: \n{sample['aspect_ner_list']}</s>"
    sample["aspect_ner_input"] = (
        f"<LM>Задача: Извлечение и классификация именованных сущностей \n{instruction}\n{sample['text']}\n"
    )
    return sample


def generate_response(
    model, tokenizer, question, top_p, temperature, prompts_path, device
):
    """function to generate response from  models"""
    with open(prompts_path, encoding="UTF-8") as fp:
        template = json.load(fp)
    num = random.randint(1, len(template))
    instruction = template["ABSA"][str(num)]
    input = f"<LM>Задача: Извлечение и классификация именованных сущностей \n{instruction}\n{question}\n"
    input_ids = tokenizer.encode(input, return_tensors="pt")
    sample_output = model.generate(
        input_ids=input_ids.to(device),
        num_beams=2,
        do_sample=True,
        max_length=400,
        top_p=top_p,
        temperature=temperature,
        top_k=70,
        early_stopping=True,
        # no_repeat_ngram_size=2,
        eos_token_id=tokenizer.eos_token_id,
    )
    out = tokenizer.decode(sample_output[0][1:], skip_special_tokens=True)
    if "</s>" in out:
        out = out[: out.find("</s>")].strip()
    return out


def prediction_to_output(model_answer, text, entity_groups):
    """function to rework prediction to the output required"""
    prediction_output = []
    entities = []
    answer = model_answer.split("Ответ:")[1].strip().replace(": ", ":")
    y_pred = set(answer.split(","))
    aspects_pred_lst = [item.split(":")[0] for item in y_pred if ":" in item]
    entity_group_lst = [item.split(":")[1] for item in y_pred if ":" in item]
    for word, entity in zip(aspects_pred_lst, entity_group_lst):
        if entity in entity_groups:
            for match in re.finditer(re.escape(word), text):
                entities.append(
                    {
                        "entity_group": entity,
                        "word": word,
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
    entities = [i for n, i in enumerate(entities) if i not in entities[:n]]
    prediction_output.append({"text": text, "entities": entities})
    return prediction_output


def tokenize_function(
    sample, tokenizer, max_source_length, max_target_length, padding="max_length"
):
    """function to tokenize data"""
    model_inputs = tokenizer(
        sample["aspect_ner_input"],
        max_length=max_source_length,
        padding=padding,
        truncation=True,
    )
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=sample["aspect_ner_output"],
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )
    # If we are padding here, replace all tokenizer.pad_token_id in the labels
    # by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
