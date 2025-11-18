import pandas as pd
import numpy as np
import re
from src.synthetic_generator.core.synthetic_dataset_generator import SyntheticDatasetGenerator

# Presumindo que SyntheticDatasetGenerator e suas dependências estão no mesmo
# contexto ou importadas corretamente.

class RulesOrchestrator:
    """
    Orquestra a geração de um dataset sintético complexo baseado em um
    conjunto de regras com interdependências de AUC e quantidade.
    """
    def __init__(self, generator: SyntheticDatasetGenerator, base_heuristic_amount=50):
        """
        :param generator: Uma instância já inicializada de SyntheticDatasetGenerator.
        :param base_heuristic_amount: O número base de sequências extras para 
                                        gerar para sub-padrões sem quantidade definida.
        """
        self.generator = generator
        self.base_heuristic_amount = base_heuristic_amount
        self._parser_regex = re.compile(r'\{([^}]+)\}|(\S+)')

    def _parse_rule_to_itemsets(self, rule_string: str) -> list[frozenset]:
        """
        Converte uma string de regra em uma lista "achatada" de frozensets.
        Gaps ([]) são ignorados para fins de verificação de subconjunto.
        
        Ex: "[A B {C D}]" -> [frozenset({'A'}), frozenset({'B'}), frozenset({'C', 'D'})]
        """
        if not rule_string:
            return []
            
        # Remove colchetes para que não sejam lidos como tokens
        cleaned_string = rule_string.replace('[', ' ').replace(']', ' ')
        
        skeleton = []
        for match in self._parser_regex.finditer(cleaned_string):
            if match.group(1): # Grupo de Itemset, ex: "C D"
                items = frozenset(match.group(1).split(' '))
                skeleton.append(items)
            elif match.group(2): # Item Único, ex: "A"
                skeleton.append(frozenset({match.group(2)}))
                
        return skeleton

    def _is_subsequence(self, skeleton_A: list, skeleton_B: list) -> bool:
        """
        Verifica se o esqueleto A é um subpadrão (subsequência) do esqueleto B.
        Usa 'issubset' para comparar itens.
        
        Ex: is_subsequence([{'B'}, {'C', 'D'}], [{'A'}, {'B'}, {'C', 'D'}]) -> True
        Ex: is_subsequence([{'D'}], [{'A'}, {'B'}, {'C', 'D'}]) -> True
        """
        it = iter(skeleton_B)
        return all(any(rule_item.issubset(seq_item) for seq_item in it) 
                   for rule_item in skeleton_A)

    def _sort_rules(self, rules_json: list) -> list:
        """
        Parseia e ordena as regras. Regras mais longas (super-padrões)
        são processadas primeiro.
        """
        parsed_rules = []
        for rule in rules_json:
            skeleton = self._parse_rule_to_itemsets(rule['element'])
            rule['_skeleton'] = skeleton
            rule['_skeleton_len'] = len(skeleton)
            parsed_rules.append(rule)
            
        # Ordena por tamanho do esqueleto, decrescente (super-padrões primeiro)
        sorted_rules = sorted(parsed_rules, key=lambda x: x['_skeleton_len'], reverse=True)
        return sorted_rules

    def generate(self, rules_json: list, n_remainder: int, global_target_auc: float, global_auc_tolerance=0.005) -> pd.DataFrame:
        """
        Gera o dataset completo orquestrando todas as regras.
        """
        
        # 1. Ordenar regras de Super-padrão para Sub-padrão
        sorted_rules = self._sort_rules(rules_json)
        
        all_generated_data = pd.DataFrame()
        
        # Mapa para rastrear contagem total (incluindo de supersets)
        rule_total_counts = {} 

        print("--- Iniciando Geração Orquestrada ---")
        
        # 2. Gerar dados para cada regra, em ordem
        for i, rule_obj in enumerate(sorted_rules):
            rule_str = rule_obj['element']
            rule_skeleton = rule_obj['_skeleton']
            target_auc = rule_obj['target_auc']
            
            print(f"\nProcessando Regra {i+1}/{len(sorted_rules)}: '{rule_str}'")
            
            # 3. Calcular quantas amostras gerar para ESTA regra (n_samples)
            
            # O 'existing_relevant_data' para balanceamento de AUC são *todos* os
            # dados gerados até agora, pois todos podem conter este sub-padrão.
            existing_data_for_auc = all_generated_data
            
            # Encontra a contagem atual deste padrão (somando todos supersets)
            current_total_count = 0
            max_superset_count = 0
            len_diff = 1
            
            for processed_rule_str, count in rule_total_counts.items():
                # Re-parseia a regra já processada para checar
                processed_skeleton = self._parse_rule_to_itemsets(processed_rule_str)
                if self._is_subsequence(rule_skeleton, processed_skeleton):
                    current_total_count += rule_generation_info[processed_rule_str]['n_generated'] # Contagem parcial
                    
                    if count > max_superset_count:
                         max_superset_count = count
                         len_diff = max(1, len(processed_skeleton) - len(rule_skeleton))

            # Se 'quantity' for definida, ela dita o N de *novas* amostras
            if 'quantity' in rule_obj:
                n_samples_to_generate = rule_obj['quantity']
                total_count_for_this_rule = current_total_count + n_samples_to_generate
            else:
                # Heurística: (50 * diff_tamanho) + contagem_max_superset
                # Garante que seja mais frequente que seus supersets.
                heuristic_boost = self.base_heuristic_amount * (len_diff + 1)
                target_total_count = max(current_total_count, max_superset_count) + heuristic_boost
                
                n_samples_to_generate = target_total_count - current_total_count
                n_samples_to_generate = max(0, n_samples_to_generate) # Não gera negativo
                total_count_for_this_rule = current_total_count + n_samples_to_generate

            print(f"  Contagem atual de supersets: {current_total_count}")
            print(f"  Novas amostras a gerar: {n_samples_to_generate} (Total alvo: {total_count_for_this_rule})")
            print(f"  Balanceando para AUC alvo combinada: {target_auc}")

            # Armazena info para regras futuras
            rule_total_counts[rule_str] = total_count_for_this_rule
            
            # (Mock) Criando um "rule_generation_info" que parece ser necessário
            if 'rule_generation_info' not in locals(): rule_generation_info = {}
            rule_generation_info[rule_str] = {'n_generated': n_samples_to_generate}


            # 4. Gerar o novo subconjunto de dados
            df_new = self.generator.generate_subset(
                rule_string=rule_str,
                n_samples=n_samples_to_generate,
                target_auc=target_auc,
                existing_relevant_data=existing_data_for_auc
            )
            
            all_generated_data = pd.concat([all_generated_data, df_new], ignore_index=True)

        # 5. Gerar o "Resto" (ruído)
        print(f"\nGerando {n_remainder} amostras de 'resto' para AUC Global {global_target_auc}")
        df_remainder = self.generator.optimize_global_dataset(
            subsets_df=all_generated_data,
            n_remainder=n_remainder,
            global_target_auc=global_target_auc,
            tolerance=global_auc_tolerance
        )
        
        all_generated_data = pd.concat([all_generated_data, df_remainder], ignore_index=True)
        
        print("\n--- Geração Orquestrada Concluída ---")
        return all_generated_data.sample(frac=1).reset_index(drop=True)