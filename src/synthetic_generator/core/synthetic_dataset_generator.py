import pandas as pd
import numpy as np
import re
import random
from scipy.stats import norm
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from src.synthetic_generator.core.sequence_generator import SequenceGenerator
from src.synthetic_generator.utils.kosarak_converter import to_kosarak_format

class SyntheticDatasetGenerator:
    def __init__(self, vocabulary, min_size=5, max_size=10, max_itemset_size=3, noise_density=1.0):
        self.vocabulary = vocabulary
        self.min_size = min_size
        self.max_size = max_size
        self.max_itemset_size = max_itemset_size
        self.noise_density = noise_density
        
        # Inicializa o gerador de regras injetando a função de criação de ruído
        self.rule_generator = SequenceGenerator(vocabulary=vocabulary, min_itemset=1, max_itemset=max_itemset_size, noise_density=noise_density)

    def _calculate_auc_scores(self, n_samples, target_auc, max_iter=50, tolerance=0.001):
        """Calcula scores de confiança para atingir uma AUC alvo (Método estático/helper)."""
        if n_samples == 0:
            return np.array([]), np.array([])
        
        mu_neg, sigma = 0.0, 1.0
        delta_mu = norm.ppf(target_auc) * np.sqrt(2 * sigma**2)
        n_pos = n_samples // 2
        n_neg = n_samples - n_pos
        targets = np.array([1] * n_pos + [0] * n_neg)

        for _ in range(max_iter):
            mu_pos = mu_neg + delta_mu
            scores_pos = np.random.normal(loc=mu_pos, scale=sigma, size=n_pos)
            scores_neg = np.random.normal(loc=mu_neg, scale=sigma, size=n_neg)
            confidence_scores = np.concatenate([scores_pos, scores_neg])
            
            if len(np.unique(confidence_scores)) < 2:
                 delta_mu += 0.1
                 continue
                 
            error = target_auc - roc_auc_score(targets, confidence_scores)
            if abs(error) < tolerance:
                break
            delta_mu += error * 1.5
            
        return targets, expit(confidence_scores)

    def generate_subset(self, rule_string, n_samples, target_auc):
        """
        Gera um DataFrame com sequências baseadas na regra e scores AUC definidos.
        """
        if n_samples <= 0:
            return pd.DataFrame(columns=['sequence', 'y_true', 'confidence'])

        targets, scaled_scores = self._calculate_auc_scores(n_samples, target_auc)
        
        if len(targets) == 0:
             return pd.DataFrame(columns=['sequence', 'y_true', 'confidence'])

        sequences_raw = []
        for _ in range(n_samples):
            seq_list = self.rule_generator.generate(rule_string, self.min_size, self.max_size)
            sequences_raw.append(seq_list)
            
        sequences_as_str = ["|".join(seq) for seq in sequences_raw]
        kosarak_sequences = [to_kosarak_format("9", s.replace("{", "").replace("}", "").split("|")) for s in sequences_as_str]

        df = pd.DataFrame({
            'sequence': kosarak_sequences, 
            'y_true': targets, 
            'confidence': scaled_scores
        })
        
        return df.sample(frac=1).reset_index(drop=True)

    def optimize_global_dataset(self, subsets_df, n_remainder, global_target_auc, tolerance=0.005):
        """
        Gera o restante dos dados (sem regra/ruído puro) para equilibrar a AUC global.
        """
        low_auc, high_auc = 0.0, 1.0
        
        initial_subset_auc = 0.5
        if not subsets_df.empty and len(np.unique(subsets_df['y_true'])) > 1:
            initial_subset_auc = roc_auc_score(subsets_df['y_true'], subsets_df['confidence'])
        
        df_best = pd.DataFrame()
        best_error = float('inf')

        for i in range(20):
            current_auc_attempt = (low_auc + high_auc) / 2
            
            df_remainder = self.generate_subset(
                rule_string="", 
                n_samples=n_remainder, 
                target_auc=current_auc_attempt
            )
            
            df_combined = pd.concat([subsets_df, df_remainder], ignore_index=True)
            
            if len(np.unique(df_combined['y_true'])) < 2:
                continue
                
            current_global = roc_auc_score(df_combined['y_true'], df_combined['confidence'])
            error = abs(current_global - global_target_auc)
            
            if error < best_error:
                best_error = error
                df_best = df_combined

            if error < tolerance:
                print(f"Convergiu na iteração {i+1}. AUC Global: {current_global:.4f}")
                return df_combined
            
            if current_global < global_target_auc:
                low_auc = current_auc_attempt
            else:
                high_auc = current_auc_attempt

        print(f"Fim das iterações. Melhor erro: {best_error:.4f}")
        return df_best