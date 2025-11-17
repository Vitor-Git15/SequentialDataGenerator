import pandas as pd
import numpy as np
import re
import random
from scipy.stats import norm
from scipy.special import expit
from sklearn.metrics import roc_auc_score

# Presumindo que suas importações locais estão corretas
from src.synthetic_generator.core.sequence_generator import SequenceGenerator
from src.synthetic_generator.utils.kosarak_converter import to_kosarak_format

class SyntheticDatasetGenerator:
    def __init__(self, vocabulary, min_size=5, max_size=10, max_itemset_size=3, noise_density=1.0):
        self.vocabulary = vocabulary
        self.min_size = min_size
        self.max_size = max_size
        self.max_itemset_size = max_itemset_size
        self.noise_density = noise_density
        
        self.rule_generator = SequenceGenerator(
            vocabulary=vocabulary, 
            min_itemset=1, 
            max_itemset=max_itemset_size, 
            noise_density=noise_density
        )

    def _calculate_auc_scores(self, n_samples, target_auc, max_iter=100, tolerance=0.005):
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
            error = target_auc - roc_auc_score(targets, confidence_scores)
            if abs(error) < tolerance:
                break
            delta_mu += error * 1.5
            
        return targets, expit(confidence_scores)

    def _generate_raw_subset(self, rule_string, n_samples, target_auc):
        """
         Gera um DataFrame *autônomo* com uma AUC alvo.
        """
        if n_samples <= 0:
            return pd.DataFrame(columns=['sequence', 'y_true', 'confidence'])

        targets, scaled_scores = self._calculate_auc_scores(n_samples, target_auc)
        
        if len(targets) == 0:
             return pd.DataFrame(columns=['sequence', 'y_true', 'confidence'])

        sequences_raw = [
            self.rule_generator.generate(rule_string, self.min_size, self.max_size) 
            for _ in range(n_samples)
        ]
            
        sequences_as_str = ["|".join(seq) for seq in sequences_raw]
        kosarak_sequences = [to_kosarak_format("9", s.replace("{ ", "").replace(" }", "").split("|")) for s in sequences_as_str]

        df = pd.DataFrame({
            'sequence': kosarak_sequences, 
            'y_true': targets, 
            'confidence': scaled_scores
        })
        
        return df.sample(frac=1).reset_index(drop=True)

    def _find_optimal_auc_for_new_data(self, existing_df, n_new_samples, combined_target_auc, rule_string_for_new, tolerance=0.005, max_iter=20):
        """
        Encontra a AUC ideal para um novo conjunto de dados para que o conjunto combinado (existente + novo) atinja a AUC global desejada.
        Retorna APENAS o novo DataFrame gerado.
        """
        low_auc, high_auc = 0.0, 1.0 # Permite AUCs < 0.5 para "piorar" a média
        
        # Define um ponto de partida para a busca
        if not existing_df.empty and len(np.unique(existing_df['y_true'])) > 1:
            initial_auc = roc_auc_score(existing_df['y_true'], existing_df['confidence'])
            if combined_target_auc < initial_auc: high_auc = initial_auc
            else: low_auc = initial_auc

        df_best_new = pd.DataFrame()
        best_error = float('inf')

        for i in range(max_iter):
            current_auc_attempt = (low_auc + high_auc) / 2
            
            df_new = self._generate_raw_subset(
                rule_string=rule_string_for_new, 
                n_samples=n_new_samples, 
                target_auc=current_auc_attempt
            )
            
            if df_new.empty: continue
            
            df_combined = pd.concat([existing_df, df_new], ignore_index=True)
            
            if len(np.unique(df_combined['y_true'])) < 2:
                if current_auc_attempt < (low_auc + 0.01): low_auc += 0.1
                elif current_auc_attempt > (high_auc - 0.01) : high_auc -= 0.1
                else: low_auc = current_auc_attempt
                continue
                
            current_global = roc_auc_score(df_combined['y_true'], df_combined['confidence'])
            error = abs(current_global - combined_target_auc)

            if error < best_error:
                best_error = error
                df_best_new = df_new

            if error < tolerance:
                print(f"Busca convergiu (Iter {i+1}). AUC Alvo: {combined_target_auc:.4f}, Global: {current_global:.4f}")
                return df_best_new
            
            if current_global < combined_target_auc:
                low_auc = current_auc_attempt
            else:
                high_auc = current_auc_attempt

        print(f"Busca (max iter). Melhor erro: {best_error:.4f} (Alvo: {combined_target_auc})")
        return df_best_new

    def generate_subset(self, rule_string, n_samples, target_auc, existing_relevant_data=None):
        """
        :param rule_string: A regra para gerar sequências.
        :param n_samples: Quantas *novas* sequências gerar para esta regra.
        :param target_auc: A AUC alvo *combinada* (existente + novas).
        :param existing_relevant_data: DataFrame contendo dados de supersets que já satisfazem esta regra.
        :return: Um DataFrame contendo *apenas* as n_samples recém-geradas.
        """
        if n_samples <= 0:
            return pd.DataFrame(columns=['sequence', 'y_true', 'confidence'])

        if existing_relevant_data is None or existing_relevant_data.empty:
            print(f"Gerando subset base para '{rule_string}' com AUC {target_auc:.4f}")
            return self._generate_raw_subset(rule_string, n_samples, target_auc)
        else:
            print(f"Gerando subset dependente para '{rule_string}' para atingir AUC {target_auc:.4f}")
            return self._find_optimal_auc_for_new_data(
                existing_df=existing_relevant_data,
                n_new_samples=n_samples,
                combined_target_auc=target_auc,
                rule_string_for_new=rule_string
            )

    def optimize_global_dataset(self, subsets_df, n_remainder, global_target_auc, tolerance=0.005):
        """
        Gera o 'resto' (ruído) para equilibrar a AUC global. Retorna apenas o DataFrame do 'resto'.
        """
        print(f"Gerando 'resto' para atingir AUC Global {global_target_auc:.4f}")
        return self._find_optimal_auc_for_new_data(
            existing_df=subsets_df,
            n_new_samples=n_remainder,
            combined_target_auc=global_target_auc,
            rule_string_for_new=""
        )