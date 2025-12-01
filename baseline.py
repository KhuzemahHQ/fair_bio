import numpy as np
from datasets import load_from_disk, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

# 1. Load Data
print("Loading dataset from local disk...")
dataset_dict = load_from_disk("./data")

train_ds = dataset_dict['train']
eval_ds = dataset_dict['test'] # Using the official test split for evaluation

# The 'profession' column is not being read as a ClassLabel, so we'll create the labels manually.
# This is a more robust way to handle it.
labels = sorted(train_ds.unique("profession"))
num_labels = len(labels)
print(f"Found {num_labels} professions: {labels}")

# Create a mapping from profession string to an integer ID
label2id = {label: i for i, label in enumerate(labels)}

def add_profession_id(example):
    return {"profession_id": label2id[example["profession"]]}

train_ds = train_ds.map(add_profession_id)
eval_ds = eval_ds.map(add_profession_id)

# 2. Preprocess Data with a Transformer Tokenizer
model_name = "distilbert-base-uncased"
print(f"\nLoading tokenizer for model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    """Tokenizes the text in the 'hard_text' column."""
    return tokenizer(examples["hard_text"], padding="max_length", truncation=True)

print("Tokenizing datasets...")
tokenized_train_ds = train_ds.map(tokenize_function, batched=True)
tokenized_eval_ds = eval_ds.map(tokenize_function, batched=True)

# The Trainer expects the label column to be named 'labels'.
tokenized_train_ds = tokenized_train_ds.rename_column("profession_id", "labels")
tokenized_eval_ds = tokenized_eval_ds.rename_column("profession_id", "labels")

# Remove columns that the model doesn't need.
# The test set has extra columns, so we find the intersection to keep.
common_columns = list(set(tokenized_train_ds.column_names) & set(tokenized_eval_ds.column_names))
columns_to_keep = ["input_ids", "attention_mask", "labels"]

train_cols_to_remove = [col for col in tokenized_train_ds.column_names if col not in columns_to_keep]
eval_cols_to_remove = [col for col in tokenized_eval_ds.column_names if col not in columns_to_keep]

tokenized_train_ds = tokenized_train_ds.remove_columns(train_cols_to_remove)
tokenized_eval_ds = tokenized_eval_ds.remove_columns(eval_cols_to_remove)

tokenized_train_ds.set_format("torch")
tokenized_eval_ds.set_format("torch")

print(f"Training set size: {len(tokenized_train_ds)}")
print(f"Evaluation set size: {len(tokenized_eval_ds)}")

# 3. Train Transformer Model
print(f"\nLoading model: {model_name}")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Define metrics for evaluation
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="test_trainer", # Directory to save model checkpoints
    do_eval=True, # Perform evaluation at the end of each epoch
    num_train_epochs=1,  # For a quick baseline, 1 epoch is fine. Increase for better performance.
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    logging_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,
    report_to="none", # Disable all integrations, including wandb

    # The arguments 'evaluation_strategy', 'save_strategy', and 'load_best_model_at_end'
    # are not supported in transformers v4.5.1. 'do_eval=True' is the legacy equivalent
    # for evaluating at the end of each epoch.
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_eval_ds,
    compute_metrics=compute_metrics,
)

# Train the model
print("\nStarting model training...")
trainer.train()

# Save the final trained model and tokenizer to a specific directory
final_model_dir = "./final_model"
print(f"\nSaving final model and tokenizer to {final_model_dir}...")
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
print("Model and tokenizer saved successfully.")

# 4. Evaluate the final model
print("\nEvaluating final model on the test set...")
eval_results = trainer.evaluate()
print(f"\nFinal Evaluation Results: {eval_results}")
print("Done.")
