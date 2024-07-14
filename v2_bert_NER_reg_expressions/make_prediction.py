import torch
from nltk.tokenize import casual_tokenize
import json
from utils import (
                    entity_groups, 
                    tokens_to_output, 
                    rework_predictions, 
                    run_regex
                )
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Load the fine-tuned model
model = AutoModelForTokenClassification.from_pretrained("./results/checkpoint-last")

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# "data/test2.json", "r") as file encoding="utf8":
with open("data/test.json", "r", encoding="utf8") as file:
    data = json.load(file)
tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint-last")

prediction_outputs = []
for item in data:
    text = item['text']
    tokens = casual_tokenize(text)
    tokenized_inputs = tokenizer(
        tokens,
        truncation=True,
        is_split_into_words=True,
        return_tensors="pt",
    )
    model.eval()
    with torch.no_grad():
        outputs = model(**tokenized_inputs.to(device))
        predictions = torch.argmax(outputs.logits, dim=2)
        tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'].tolist()[0])
        prediction_output = tokens_to_output(
            tokens=tokens,
            preds=predictions.tolist()[0],
            text=text,
            entity_groups=entity_groups,
        )
        reworked_prediction = rework_predictions(prediction_output[0])
        regex_applied_prediction = run_regex(reworked_prediction)
        prediction_outputs.append(regex_applied_prediction)

# Save the predictions_output to a JSON file with UTF-8 encoding
output_file = "data/submission.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(prediction_outputs, f, ensure_ascii=False, indent=4)

print(f"Predictions saved to {output_file}")
