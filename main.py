# main.py
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from src.synthetic_generator.utils.file_handler import load_signal_rules, load_vocabulary
from src.synthetic_generator.utils.vocabulary_handler import get_clean_vocabulary
from src.synthetic_generator.core.synthetic_dataset_generator import SyntheticDatasetGenerator
from src.synthetic_generator.utils.verification import verify_and_print_results
from src.synthetic_generator.utils.parser import validate_parser_input

def parse_arguments():
    parser = argparse.ArgumentParser(description="Gerador de sequências com AUC alvo")
    parser.add_argument("--gauc", type=float, required=True, help="Target AUC para o conjunto final")
    parser.add_argument("--voc", type=str, required=True, help="Caminho para o vocabulário")
    parser.add_argument("--sig", type=str, required=True, help="Caminho para o arquivo de regras")
    parser.add_argument("--gen-itemsets", action='store_true', help="Se especificado, gera itemsets aleatórios no ruído de fundo.")
    parser.add_argument("--noise-density", type=float, default=1.0, help="Densidade do ruído (0.0 a 1.0). 1.0 = denso (padrões aleatórios se formam). 0.0 = esparso (itens de ruído são únicos).")
    parser.add_argument("--maxseq", type=int, default=1000, help="Número máximo de sequências a serem geradas no conjunto final.")
    return parser.parse_args()

def save_sequences_to_file(sequences, filename="data/synth.dat"):
    np.savetxt(filename, sequences, fmt="%s")
    print(f"Sequências salvas em '{filename}'")

def save_dataframe_to_csv(df, filename="data/synth.csv"):
    df.to_csv(filename, index=False)
    print(f"DataFrame salvo em '{filename}'")

def main():
    args = parse_arguments()

    if not 0.0 <= args.noise_density <= 1.0:
        args.noise_density = 1.0

    target_global_auc = args.gauc
    vocabulary = load_vocabulary(args.voc)
    signal_rules = load_signal_rules(args.sig)

    for rule in signal_rules:
        is_valid, message = validate_parser_input(rule['element'])
        if not is_valid:
            raise ValueError(f"Regra inválida para o elemento '{rule['element']}': {message}")
    
    CLEANED_VOCABULARY = get_clean_vocabulary(vocabulary, signal_rules)
    synthetic_dataset_generator = SyntheticDatasetGenerator(
        vocabulary=CLEANED_VOCABULARY,
        noise_density=args.noise_density,
        max_itemset_size=3 if args.gen_itemsets else 1
    )
    df_list = []
    for rule in signal_rules:
        element, quantity, target_auc = rule['element'], rule['quantity'], rule['target_auc']
        
        print(f"Gerando subconjunto com elemento '{element}' e AUC alvo de {target_auc}...")
        df_rule = synthetic_dataset_generator.generate_subset(
            rule_string=element,
            n_samples=quantity,
            target_auc=target_auc,
        )
        
        auc_actual = roc_auc_score(df_rule['y_true'], df_rule['confidence'])
        print(f"AUC real do subconjunto com '{element}': {auc_actual:.4f}\n")
        df_list.append(df_rule)
    print(df_list)
    df_rules_concatenated = pd.concat(df_list, ignore_index=True)

    df_final = synthetic_dataset_generator.optimize_global_dataset(
        df_rules_concatenated,
        n_remainder=args.maxseq - len(df_rules_concatenated),
        global_target_auc=target_global_auc
    )
    print(f"\nDataset final gerado com {len(df_final)} sequências.")
    print(f"df_final['y_true'] unique values: {df_final['y_true'].unique()}")    
    # --- 4. Verificação e Salvamento ---
    # verify_and_print_results(df_final, signal_rules, GLOBAL_AUC_X)
    sequences = df_final['sequence']
    np.savetxt("data/synth.dat", sequences, fmt="%s")
    # print("\nSequências salvas em 'data/synth.dat'")
    df_final.to_csv("data/synth.csv", index=False)
    # print("Dataset final salvo em 'data/synth.csv'")

if __name__ == "__main__":
    main()