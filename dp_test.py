import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

def test_demographic_parity():
    """
    Loads a fine-tuned model and evaluates its demographic parity on the test set
    with respect to gender.
    """

    # Select Model : pellement99/distilbert-occupation-classifier, pellement99/occupation-classification-synthetic, or pellement99/occupation-classification-very-synthetic
    MODEL_ID = "pellement99/pellement99/occupation-classification-very-synthetic" 
    DATA_DIR = "./data"
    PREDICTIONS_FILE = "very_predictions.csv"

    # 1. Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device} ---")

    # 3. Load Data
    print(f"Loading test data from '{DATA_DIR}'...")
    try:
        dataset_dict = load_from_disk(DATA_DIR)
        test_ds = dataset_dict['test']
    except FileNotFoundError:
        print(f"Error: Dataset not found at '{DATA_DIR}'. Please run baseline.py to download it first.")
        return

    # Use the official label mapping provided by the dataset creators.
    id2label = {
        0: 'accountant', 1: 'architect', 2: 'attorney', 3: 'chiropractor',
        4: 'comedian', 5: 'composer', 6: 'dentist', 7: 'dietitian',
        8: 'dj', 9: 'filmmaker', 10: 'interior_designer', 11: 'journalist',
        12: 'model', 13: 'nurse', 14: 'painter', 15: 'paralegal',
        16: 'pastor', 17: 'personal_trainer', 18: 'photographer',
        19: 'physician', 20: 'poet', 21: 'professor', 22: 'psychologist',
        23: 'rapper', 24: 'software_engineer', 25: 'surgeon',
        26: 'teacher', 27: 'yoga_teacher'
    }
    
    
    # 4. Generate or Load Predictions
    if os.path.exists(PREDICTIONS_FILE):
        print(f"Loading existing predictions from '{PREDICTIONS_FILE}'...")
        df = pd.read_csv(PREDICTIONS_FILE)
    else:
        print("Generating predictions for the test set (this will happen only once)...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
            model.to(device)
            model.eval()
        except OSError:
            print(f"Error: Could not download model '{MODEL_ID}'. Check model name and internet connection.")
            return

        # Filter out bad data before prediction. The dataset uses integer labels for gender.
        test_ds = test_ds.filter(lambda x: x['hard_text'] is not None and x['gender'] is not None)

        results = []
        with torch.no_grad():
            for example in tqdm(test_ds, desc="Predicting"):
                inputs = tokenizer(example["hard_text"], return_tensors="pt", truncation=True, padding=True).to(device)
                logits = model(**inputs).logits
                prediction = torch.argmax(logits, dim=-1).item()
                
                results.append({
                    "gender_id": example["gender"],
                    "predicted_id": prediction
                })
        df = pd.DataFrame(results)
        df.to_csv(PREDICTIONS_FILE, index=False)
        print(f"Predictions saved to '{PREDICTIONS_FILE}'.")


    # 5. Calculate Demographic Parity
    print("\n--- Overall Demographic Parity Evaluation ---")

    df['predicted_label'] = df['predicted_id'].map(id2label)


    # Separate by gender ID. In this dataset: 0 = female, 1 = male.
    males = df[df['gender_id'] == 0]
    females = df[df['gender_id'] == 1]

    # Calculate the distribution of predicted professions for each gender
    male_dist = males['predicted_label'].value_counts(normalize=True)
    female_dist = females['predicted_label'].value_counts(normalize=True)

    # Combine into a single DataFrame for plotting
    dist_df = pd.DataFrame({'Male': male_dist, 'Female': female_dist}).fillna(0)
    dist_df['Difference'] = abs(dist_df['Male'] - dist_df['Female'])
    dist_df = dist_df.sort_values(by='Difference', ascending=False)

    print("Top 10 professions with the largest demographic parity difference:")
    print(dist_df.head(10))

    # 6. Visualize the distributions
    print("\nGenerating plot of prediction distributions by gender...")
    plot_df = dist_df.drop(columns=['Difference']).sort_index() # Sort alphabetically for the plot
    plot_df.plot(kind='bar', figsize=(18, 8), width=0.8)

    plt.title('Distribution of Predicted Professions by Gender')
    plt.ylabel('Proportion of Predictions')
    plt.xlabel('Profession ID')
    plt.xticks(rotation=90)
    plt.ylim(0, 0.35)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("results/very_synth_parity_440pm.png")

if __name__ == "__main__":
    test_demographic_parity()
