# main.py
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from src.synthetic_generator.utils.validator import SequenceValidator
from src.synthetic_generator.utils.parser import validate_rules
from src.synthetic_generator.utils.vocabulary_handler import get_clean_vocabulary
from src.synthetic_generator.utils.file_handler import load_signal_rules, load_vocabulary, save_sequences_to_file, save_dataframe_to_csv
from src.synthetic_generator.core.synthetic_dataset_generator import SyntheticDatasetGenerator

def parse_arguments():
    parser = argparse.ArgumentParser(description="Gerador de sequências com AUC alvo")
    parser.add_argument("--gauc", type=float, required=True, help="Target AUC para o conjunto final")
    parser.add_argument("--voc", type=str, required=True, help="Caminho para o vocabulário")
    parser.add_argument("--sig", type=str, required=True, help="Caminho para o arquivo de regras")
    parser.add_argument("--gen-itemsets", action='store_true', help="Se especificado, gera itemsets aleatórios no ruído de fundo.")
    parser.add_argument("--noise-density", type=float, default=1.0, help="Densidade do ruído (0.0 a 1.0). 1.0 = denso (padrões aleatórios se formam). 0.0 = esparso (itens de ruído são únicos).")
    parser.add_argument("--maxseq", type=int, default=10, help="Número máximo de sequências a serem geradas no conjunto final.")
    return parser.parse_args()

def main():
    args = parse_arguments()

    if not 0.0 <= args.noise_density <= 1.0:
        args.noise_density = 1.0

    target_global_auc = args.gauc
    vocabulary = load_vocabulary(args.voc)
    signal_rules = load_signal_rules(args.sig)
    validate_rules(signal_rules)
    
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

    df_remainder = synthetic_dataset_generator.optimize_global_dataset(
        df_rules_concatenated,
        n_remainder=args.maxseq - len(df_rules_concatenated),
        global_target_auc=target_global_auc
    )
    df_final = pd.concat([df_rules_concatenated, df_remainder], ignore_index=True)
    validator = SequenceValidator()

    validator.validate(df_final, signal_rules)

    save_sequences_to_file(df_final['sequence'].tolist(), filename="data/synth.dat")
    save_dataframe_to_csv(df_final, filename="data/synth.csv")

if __name__ == "__main__":
    main()