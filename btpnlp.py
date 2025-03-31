!pip install transformers datasets scikit-learn

import json
import csv

def label_tatqa_questions(input_file, output_csv):
    """
    Reads a TAT-QA JSON file, extracts each question,
    labels it 'financial' if answer_type is 'arithmetic' (or contains certain keywords),
    otherwise 'logical', and writes to output_csv.
    """
    print(f"Debug: Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    financial_keywords = ["revenue", "sum", "total", "%", "percent", "million", "billion"]

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["question", "label"])

        count_questions = 0
        for entry in data:
            questions_list = entry.get("questions", [])
            for q in questions_list:
                q_text = q["question"].strip()
                answer_type = q.get("answer_type", "").lower()


                if answer_type == "arithmetic":
                    label = "financial"
                else:

                    q_lower = q_text.lower()
                    if any(kw in q_lower for kw in financial_keywords):
                        label = "financial"
                    else:
                        label = "logical"

                writer.writerow([q_text, label])
                count_questions += 1

        print(f"Debug: Wrote {count_questions} questions to {output_csv}.")
label_tatqa_questions("/content/tatqa_dataset_train.json", "/tatqa_labeled_train.csv")
label_tatqa_questions("/content/tatqa_dataset_test.json",   "/tatqa_labeled_dev.csv")
label_tatqa_questions("/content/tatqa_dataset_test.json",  "/tatqa_labeled_test.csv")

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

print("Debug: Loading labeled CSV files for TAT-QA...")

train_df = pd.read_csv("/tatqa_labeled_train.csv")
dev_df   = pd.read_csv("/tatqa_labeled_dev.csv")
test_df  = pd.read_csv("/tatqa_labeled_test.csv")

print("Debug: Shapes => Train:", train_df.shape, "Dev:", dev_df.shape, "Test:", test_df.shape)
print("Debug: Example train:\n", train_df.head())

label2id = {"financial": 0, "logical": 1}
id2label = {0: "financial", 1: "logical"}

train_df['label_id'] = train_df['label'].map(label2id)
dev_df['label_id']   = dev_df['label'].map(label2id)
test_df['label_id']  = test_df['label'].map(label2id)

!pip install -q google-generativeai

import google.generativeai as genai

genai.configure(api_key="AIzaSyAyKwyoOkNevA8d5v_yscDV0ZVnxOO9JO0")

model = genai.GenerativeModel(model_name="gemini-1.5-flash")

class TatQADataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'text': row['question'],
            'label': row['label']
        }

train_df = pd.read_csv("/tatqa_labeled_train.csv")
dev_df   = pd.read_csv("/tatqa_labeled_dev.csv")
test_df  = pd.read_csv("/tatqa_labeled_test.csv")

train_dataset = TatQADataset(train_df)
dev_dataset   = TatQADataset(dev_df)
test_dataset  = TatQADataset(test_df)

BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader   = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Debug: Data preparation complete. Using raw text for Gemini.")

import time

def classify_with_gemini(question):
    """
    Sends a question to Gemini 1.5 Flash and returns the predicted label.
    """
    prompt = f"""
    Classify the following question into one of two categories:
    - 'financial' (if it involves arithmetic, calculations, or financial terms)
    - 'logical' (if it is about comparisons, reasoning, or non-mathematical analysis)

    Question: "{question}"

    Respond ONLY with 'financial' or 'logical'.
    """

    try:
        response = model.generate_content(prompt)
        gemini_label = response.text.strip().lower()

        print(f"🔹 Debug: Question='{question}' | Gemini Output='{gemini_label}'")

        if gemini_label not in ["financial", "logical"]:
            print(f"Unexpected response: '{gemini_label}', defaulting to 'logical'")
            return "logical"
        return gemini_label

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "logical"

sample_question = test_df.iloc[0]["question"]
print("🔹 Sample classification:", classify_with_gemini(sample_question))

def evaluate_gemini_model(dataloader):
    """
    Runs the Gemini model on the dataset and evaluates accuracy.
    """
    all_preds = []
    all_labels = []
    total_count = 0
    correct_count = 0

    for batch in dataloader:
        for text, label in zip(batch["text"], batch["label"]):
            gemini_pred = classify_with_gemini(text)
            all_preds.append(gemini_pred)
            all_labels.append(label)

            if gemini_pred != label:
                print(f"Debug: Mismatch! Q='{text}' | True='{label}' | Predicted='{gemini_pred}'")

            total_count += 1
            correct_count += (gemini_pred == label)

            time.sleep(0.5)

    accuracy = correct_count / total_count * 100
    print(f"\nDebug: Gemini Accuracy = {accuracy:.2f}%")

    print("\nGemini Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["financial", "logical"]))

evaluate_gemini_model(test_loader)