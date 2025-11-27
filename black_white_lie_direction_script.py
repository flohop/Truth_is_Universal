import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch

def accuracy_of_lr(model, tokenizer, model_family: str, layer: int, figure_output_dir: str, working_dir: str, run_neutral):


    # 1. Configuration
    # =================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe_path = os.path.join(working_dir, f'black_white_classification_{model_family}.pt')
    text_column = "statement" # Ensure your CSVs have this column header

    # UPDATED PATHS
    black_lie_files = [
        "colleagues",
        "police",
        "sales",
        "teachers"
    ]

    black_lie_files = ["datasets/lie_statements/black_lies/" + file + ".csv" for file in black_lie_files]

    if run_neutral:
        white_lie_files = ["neutral"]
    else:
        white_lie_files = [
            "colleagues",
            "friendship",
            "parents",
            "teachers"
        ]

    white_lie_files = ["datasets/lie_statements/white_lies/" + file + ".csv" for file in white_lie_files]

    # 2. Helper Functions
    # ===================
    def load_datasets_from_list(file_list, label_int, text_col):
        data_entries = []
        for file_path in file_list:
            if not os.path.exists(file_path): 
                print(f"Warning: File not found: {file_path}")
                continue
            try:
                df = pd.read_csv(file_path)
                # Strip whitespace from column names just in case
                df.columns = [c.strip() for c in df.columns]
                
                if text_col in df.columns:
                    for text in df[text_col].dropna():
                        data_entries.append((str(text), label_int))
                    print(f"Loaded {len(df)} rows from {file_path}")
                else:
                    print(f"Error: Column '{text_col}' not found in {file_path}. Found: {df.columns.tolist()}")
            except Exception as e: 
                print(f"Error reading {file_path}: {e}")
        return data_entries

    def get_live_activation(text, model, tokenizer, layer_idx, device):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[layer_idx][0, -1, :]

    # 3. Collect Scores
    # =================
    # Load Probe
    d_probe = torch.load(probe_path, map_location=device)
    d_probe = d_probe.to(device)
    print(f"Probe loaded. Shape: {d_probe.shape}, Dtype: {d_probe.dtype}")

    # Load Data
    data = []
    data.extend(load_datasets_from_list(black_lie_files, 1, text_column)) # Black = 1
    data.extend(load_datasets_from_list(white_lie_files, 0, text_column)) # White = 0

    if len(data) == 0:
        raise ValueError("No data loaded. Please check your file paths and column names.")

    all_scores = []
    all_labels = []

    print(f"\nRunning inference on {len(data)} statements...")

    for i, (text, label) in enumerate(data):
        # Get Activation
        act = get_live_activation(text, model, tokenizer, layer, device)
        
        # Fix dtype mismatch (BFloat16 vs Float32)
        act = act.to(dtype=d_probe.dtype) 
        
        # Project
        score = torch.dot(act, d_probe).item()
        
        all_scores.append(score)
        all_labels.append(label)
        
        if i % 50 == 0: print(f"Processed {i}/{len(data)}...")

    # 4. Find Optimal Threshold
    # =========================
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]

    print(f"\n=== CALIBRATION RESULTS ===")
    print(f"Optimal Threshold Found: {best_thresh:.4f}")

    # 5. Visualization
    # ================
    plt.figure(figsize=(10, 6))
    scores_black = [s for s, l in zip(all_scores, all_labels) if l == 1]
    scores_white = [s for s, l in zip(all_scores, all_labels) if l == 0]

    plt.hist(scores_black, bins=20, alpha=0.6, label='Black Lies (Class 1)', color='black')
    plt.hist(scores_white, bins=20, alpha=0.6, label="Neutral Statements (Class 0)" if run_neutral else 'White Lies (Class 0)', color='skyblue')
    plt.axvline(best_thresh, color='red', linestyle='dashed', linewidth=2, label=f'Optimal ({best_thresh:.2f})')
    plt.axvline(0, color='gray', linestyle='dotted', linewidth=1, label='Zero (Original)')

    plt.title(f'Probe Score Distribution (Layer {layer})')
    plt.xlabel('Dot Product Score')
    plt.ylabel('Count')
    plt.legend()

    if figure_output_dir:
        file_name = "black_neutral_probe_accuracy.png" if run_neutral else "black_white_probe_accuracy.png"
        plt.savefig(os.path.join(figure_output_dir, file_name), dpi=300, bbox_inches="tight")

    # 6. Recalculate Accuracy
    # =======================
    predictions = [1 if s > best_thresh else 0 for s in all_scores]

    black_correct = sum([1 for p, l in zip(predictions, all_labels) if p == l and l == 1])
    black_total = sum([1 for l in all_labels if l == 1])
    white_correct = sum([1 for p, l in zip(predictions, all_labels) if p == l and l == 0])
    white_total = sum([1 for l in all_labels if l == 0])

    results_data = {
        "threshold": float(best_thresh),
        "black_lies": {
            "correct": int(black_correct),
            "total": int(black_total),
            "accuracy_percentage": float(black_correct / black_total * 100)
        },
        "white_lies": {
            "correct": int(white_correct),
            "total": int(white_total),
            "accuracy_percentage": float(white_correct / white_total * 100)
        },
        "overall_accuracy_percentage": float(accuracy_score(all_labels, predictions) * 100)
    }

    print(f"\n=== Final Accuracy with Threshold {best_thresh:.4f} ===")
    print(f"Black Lies Accuracy: {black_correct}/{black_total} ({black_correct/black_total*100:.2f}%)")
    print(f"White Lies Accuracy: {white_correct}/{white_total} ({white_correct/white_total*100:.2f}%)")
    print(f"Overall Accuracy:    {accuracy_score(all_labels, predictions)*100:.2f}%")

    if figure_output_dir:
        with open(os.path.join(figure_output_dir, "b_n_accuract_threshold.json" if run_neutral else"b_w_accuracy_threshold.json"), "w") as f:
            json.dump(results_data, f, indent=4)