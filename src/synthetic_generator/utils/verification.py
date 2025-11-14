# src/verification.py
import re
import pandas as pd
from sklearn.metrics import roc_auc_score

def verify_and_print_results(df_final, signal_rules, global_target_auc):
    """
    Calcula e imprime a AUC global e a AUC de cada subconjunto definido nas regras.
    """
    print("\n--- Verificação Final ---")
    final_global_auc = roc_auc_score(df_final['y_true'], df_final['confidence'])
    print(f"AUC Global Alvo: {global_target_auc}")
    print(f"AUC Global Final: {final_global_auc:.4f}")

    for rule in signal_rules:
        element = rule['element']
        target_auc = rule['target_auc']
        df_subset = pd.DataFrame()

        if element.startswith('[') and element.endswith(']'):
            content = element.strip('[]')
            pattern = f" {re.escape(content)} -1"
            label_pattern = f"^{re.escape(df_final['sequence'].iloc[0].split()[0])} {re.escape(content)} -1"
            df_subset = df_final[
                df_final['sequence'].str.contains(pattern) | df_final['sequence'].str.contains(label_pattern)
            ]

        elif ' ' in element:
            pattern = '.*'.join([re.escape(item) for item in element.split(' ')])
            df_subset = df_final[df_final['sequence'].str.contains(pattern, regex=True)]
        
        else:
            pattern = f" {re.escape(element)} -1"
            label_pattern = f"^{re.escape(df_final['sequence'].iloc[0].split()[0])} {re.escape(element)} -1"
            df_subset = df_final[
                df_final['sequence'].str.contains(pattern) | df_final['sequence'].str.contains(label_pattern)
            ]

        if not df_subset.empty:
            final_subset_auc = roc_auc_score(df_subset['y_true'], df_subset['confidence'])
            print(f"\n> Subconjunto '{element}' (Alvo AUC: {target_auc})")
            print(f"  - AUC Final Verificada: {final_subset_auc:.4f}")
            print(f"  - Número de sequências: {len(df_subset)}")
        else:
            print(f"\nAVISO: Nenhum subconjunto encontrado para o elemento '{element}' na verificação final.")