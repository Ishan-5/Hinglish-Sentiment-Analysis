import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Config
MODEL_NAME = "google/muril-base-cased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5  # Synced below with TrainingArguments
OUTPUT_DIR = "../models/muril-hinglish"

# Label mapping
label2id = {"Negative": 0, "Positive": 1, "Neutral": 2}
id2label = {0: "Negative", 1: "Positive", 2: "Neutral"}

# Load dataset
dataset = load_dataset("shae2977/hinglish-youtube-sentiments-dataset")
df_train = dataset['train'].to_pandas()

# Drop unused columns
df_train = df_train[['comment', 'sentiment']].dropna()
df_train['label'] = df_train['sentiment'].map(label2id)

# Split manually 80/20
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['label'])

print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# --- CHANGE 1: Calculate Class Weights for Imbalance ---
classes = np.unique(train_df['label'].values)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_df['label'].values)
class_weights = torch.tensor(weights, dtype=torch.float)

# Convert to HuggingFace Dataset
from datasets import Dataset
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch['comment'], truncation=True, padding='max_length', max_length=MAX_LEN)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("label", "labels")
val_ds = val_ds.rename_column("label", "labels")

train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# --- CHANGE 2: Custom Trainer to inject Imbalance Weights ---
class ImbalancedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Move weights to the active GPU/CPU device dynamically
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Metrics
# --- CHANGE 3: Track Macro F1 for unbiased evaluation ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    return {"macro_f1": f1_macro, "weighted_f1": f1_weighted}

# Training args
# --- CHANGE 4: Performance improvements and setting limits ---
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,                       # Fixed variable consistency mismatch
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",              # Optimize for unrepresented classes
    save_total_limit=2,                            # Keeps disk from getting cluttered with shards
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=True,
    report_to="none"
)

# Trainer
trainer = ImbalancedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training complete. Model saved.")

# Final evaluation
preds_output = trainer.predict(val_ds)
preds = np.argmax(preds_output.predictions, axis=1)
labels = preds_output.label_ids
print(classification_report(labels, preds, target_names=["Negative", "Positive", "Neutral"]))
