from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset, DatasetDict
import pandas as pd
import evaluate
from utils import (
    preprocess_text,
    tokenize_and_align_labels,
    entity_groups,
    prepare_compute_metrics
)

# Set seed for reproducibility
seed = 139

data = pd.read_json("data/train.json")
data = data.dropna()
dataset = Dataset.from_pandas(data)
# dataset = dataset.shuffle()
dataset_train_test = dataset.train_test_split(test_size=0.01, seed=seed)
final_ds = DatasetDict(
        {
            "train": dataset_train_test["train"],
            "test": dataset_train_test["test"],
        }
    )
processed_dataset = final_ds.map(
    preprocess_text,
    fn_kwargs={
        "entity_groups": entity_groups,
    },
    remove_columns=["text", "entities"],
)

entity_groups = entity_groups
entity_groups.insert(0, "O")
id2label = {i: label for i, label in enumerate(entity_groups)}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
label_list = list(label2id.keys())

model_id = "ai-forever/ruBert-base"

tokenizer = AutoTokenizer.from_pretrained(
    model_id
)

tokenized_dataset = processed_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    fn_kwargs={
        "tokenizer": tokenizer,
    },
    remove_columns=["tokens", "ner_tag"],
)

seqeval = evaluate.load("seqeval")

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

model = AutoModelForTokenClassification.from_pretrained(
    model_id, num_labels=num_labels, id2label=id2label, label2id=label2id
)

compute_metrics = prepare_compute_metrics(label_list=label_list,
                                          seqeval=seqeval)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=5e-05,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.1,
    evaluation_strategy="epoch",
    push_to_hub=False,
    save_strategy="no",
    group_by_length=True,
    warmup_ratio=0.1,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save the fine-tuned model locally
model.save_pretrained("./results/checkpoint-last")
tokenizer.save_pretrained("./results/checkpoint-last")
