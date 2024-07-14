import torch
import json
from utils import (
                    entity_groups,
                    prediction_to_output,
                    generate_response

                )
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the fine-tuned model
model = AutoModelForSeq2SeqLM.from_pretrained("./results/checkpoint-last")

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# "data/test2.json", "r") as file encoding="utf8":
with open("data/test.json", "r", encoding="utf8") as file:
    data = json.load(file)
tokenizer = AutoTokenizer.from_pretrained("./results/checkpoint-last",
                                          eos_token="</s>")

prediction_outputs = []
for item in data:
    text = item['text']
    prediction = generate_response(
        model=model,
        tokenizer=tokenizer,
        question=text,
        top_p=0.6,
        temperature=0.7,
        prompts_path='prompts.json',
        device=device
    )
    reworked_prediction = prediction_to_output(
        model_answer=prediction,
        text=text,
        entity_groups=entity_groups
    )

    prediction_outputs.append(reworked_prediction[0])

# Save the predictions_output to a JSON file with UTF-8 encoding
output_file = "data/submission.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(prediction_outputs, f, ensure_ascii=False, indent=4)

print(f"Predictions saved to {output_file}")
