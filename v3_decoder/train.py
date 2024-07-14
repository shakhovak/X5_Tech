from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import Dataset, DatasetDict, concatenate_datasets
import pandas as pd
from utils import (
    createATE_dataset,
    createASC_dataset,
    createABSA_dataset, 
    tokenize_function
)
warnings.filterwarnings("ignore")


# Data augmentation through splitting task in 3 and training the model together on all tasks
# additionally data is augmented by using different instructions to each dataset

data = pd.read_json("data/train.json")
data = data.drop_duplicates(subset="text")

dataset = Dataset.from_pandas(data)
dataset = dataset.shuffle()
train_test_split = dataset.train_test_split(test_size=0.1)

ATE_dataset_1 = train_test_split["train"].map(
    createATE_dataset, fn_kwargs={"prompts_file_path": "prompts.json"}
)
ATE_dataset_2 = train_test_split["train"].map(
    createATE_dataset, fn_kwargs={"prompts_file_path": "prompts.json"}
)
ASC_dataset_1 = train_test_split["train"].map(
    createASC_dataset, fn_kwargs={"prompts_file_path": "prompts.json"}
)
ASC_dataset_2 = train_test_split["train"].map(
    createASC_dataset, fn_kwargs={"prompts_file_path": "prompts.json"}
)
ABSA_dataset_1 = train_test_split["train"].map(
    createABSA_dataset, fn_kwargs={"prompts_file_path": "prompts.json"}
)
ABSA_dataset_2 = train_test_split["train"].map(
    createABSA_dataset, fn_kwargs={"prompts_file_path": "prompts.json"}
)
ABSA_test = train_test_split["test"].map(
    createABSA_dataset, fn_kwargs={"prompts_file_path": "prompts.json"}
)

combined_datasets = concatenate_datasets(
    [
        ATE_dataset_1,
        ASC_dataset_1,
        ABSA_dataset_1,
        ATE_dataset_2,
        ASC_dataset_2,
        ABSA_dataset_2,
    ],
)
combined_datasets = combined_datasets.shuffle()
dataset_train_test = combined_datasets.train_test_split(test_size=0.1)

final_ds = DatasetDict(
    {
        "train": dataset_train_test["train"],
        "test": dataset_train_test["test"],
        "val": ABSA_test,
    }
)

tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRED-T5-large", 
                                          eos_token="</s>")

# defining max length for the model
tokenized_inputs = concatenate_datasets(
    [
        final_ds["train"],
        final_ds["test"],
        final_ds["val"],
    ],
).map(
    lambda x: tokenizer(x["aspect_ner_input"], truncation=True),
    batched=True,
    remove_columns=[
        "text",
        "entities",
        "aspect_list",
        "ner_list",
        "aspect_ner_list",
        "aspect_ner_output",
        "aspect_ner_input",
        "__index_level_0__",
    ],
)
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])

tokenized_targets = concatenate_datasets(
    [
        final_ds["train"],
        final_ds["test"],
        final_ds["val"],
    ],
).map(
    lambda x: tokenizer(x["aspect_ner_output"], truncation=True),
    batched=True,
    remove_columns=[
        "text",
        "entities",
        "aspect_list",
        "ner_list",
        "aspect_ner_list",
        "aspect_ner_output",
        "aspect_ner_input",
        "__index_level_0__",
    ],
)
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])

# final tokenization of the dataset
tokenized_dataset = final_ds.map(
    tokenize_function,
    batched=True,
    fn_kwargs={
        "max_source_length": max_source_length,
        "max_target_length": max_target_length,
        "tokenizer": tokenizer,
    },
    remove_columns=[
        "text",
        "entities",
        "aspect_list",
        "ner_list",
        "aspect_ner_list",
        "aspect_ner_output",
        "aspect_ner_input",
        "__index_level_0__",
    ],
)

model = AutoModelForSeq2SeqLM.from_pretrained("ai-forever/FRED-T5-large")
# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8,
)

repository_id = f"fred-test"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=repository_id,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False,  # Overflows with fp16
    learning_rate=1e-4,
    num_train_epochs=10,
    warmup_ratio=0.1,
    weight_decay=0.01,
    optim="adamw_torch",
    # logging & evaluation strategies
    evaluation_strategy="epoch",
    save_strategy="no",
    save_total_limit=1,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)
trainer.train()

# Save the fine-tuned model locally
model.save_pretrained("./results/checkpoint-last")
tokenizer.save_pretrained("./results/checkpoint-last")
