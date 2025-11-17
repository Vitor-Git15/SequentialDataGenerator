# src/file_handler.py

import json
from typing import List, Dict
import numpy as np

def load_signal_rules(file_path: str) -> List[Dict]:
    """Load and validate signaling rules from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)

        if not isinstance(rules, list):
            print(f"Format Error: The content of '{file_path}' must be a JSON list ([...]).")
            return []

        required_keys = {"element", "quantity", "target_auc"}
        for i, rule in enumerate(rules):
            if not isinstance(rule, dict) or not required_keys.issubset(rule.keys()):
                print(f"Format Error: Item {i} in '{file_path}' does not contain the required keys: {required_keys}.")
                return []

        rules.sort(key=lambda x: len(x["element"].split(" ")), reverse=True)

        return rules
    
    except Exception as e:
        print(f"An unexpected error occurred while loading '{file_path}': {e}")
        return []
    
def load_vocabulary(file_path):
    """Load vocabulary from a text file."""
    try:
        with open(file_path, 'r') as f:
            vocab = [line.strip() for line in f if line.strip()]
        return vocab
    except Exception as e:
        print(f"Error: Could not load vocabulary from '{file_path}'. Using default vocabulary.")
        return ['T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    

def save_sequences_to_file(sequences, filename="data/synth.dat"):
    np.savetxt(filename, sequences, fmt="%s")
    print(f"Sequências salvas em '{filename}'")

def save_dataframe_to_csv(df, filename="data/synth.csv"):
    df.to_csv(filename, index=False)
    print(f"DataFrame salvo em '{filename}'")