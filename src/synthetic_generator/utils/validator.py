import re
from wsgiref import validate
import numpy as np
import functools # Usado para cache/memoization
from sklearn.metrics import roc_auc_score

class SequenceValidator:
    """
    Valida sequências no formato Kosarak contra uma regra ESTRITA 
    que entende a sintaxe de gap [...]
    """

    def _parse_kosarak_to_sets(self, kosarak_str: str) -> list[frozenset]:
        """
        Converte uma string de sequência Kosarak em uma lista de frozensets.
        """
        cleaned = re.sub(r'^\s*9\s+', '', kosarak_str)
        cleaned = re.sub(r'\s+-2\s*$', '', cleaned).strip()
        if not cleaned:
            return []
        
        transactions = re.split(r'\s+-1\s+', cleaned) 
        
        seq_sets = []
        for trans in transactions:
            if not trans.strip():
                continue
            items = re.split(r'\s+', trans.strip())
            seq_sets.append(frozenset(items))
            
        return seq_sets

    def _parse_rule_to_skeleton(self, rule_string: str) -> list:
        """
        Converte a regra em um esqueleto que inclui marcadores de 'GAP'.
        
        Ex: "A [B C] D" -> [ {'A'}, {'B'}, 'GAP', {'C'}, {'D'} ]
        Ex: "A B C"     -> [ {'A'}, {'B'}, {'C'} ]
        Ex: "[A B C]"   -> [ {'A'}, 'GAP', {'B'}, 'GAP', {'C'} ]
        """
        if not rule_string:
            return []
            
        tokens = re.findall(r'\[|\]|\{|\}|[a-zA-Z0-9_]+', rule_string)
        
        skeleton = []
        inside_gap_group = False
        last_item_was_in_gap = False
        
        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token == '[':
                inside_gap_group = True
                i += 1
                continue
            elif token == ']':
                inside_gap_group = False
                last_item_was_in_gap = False 
                i += 1
                continue
            if last_item_was_in_gap and inside_gap_group:
                skeleton.append("GAP")
                
            current_itemset = set()
            if token == '{':
                i += 1
                while i < len(tokens) and tokens[i] != '}':
                    current_itemset.add(tokens[i])
                    i += 1
                i += 1 
            else:
                current_itemset.add(token)
                i += 1
            
            skeleton.append(frozenset(current_itemset))
            last_item_was_in_gap = inside_gap_group
                
        return skeleton
    
    @functools.lru_cache(maxsize=None) 
    def _match_recursive(self, rule_ptr, seq_ptr, rule_skeleton, seq_sets):
        """
        Tenta dar match no resto da 'rule_skeleton' (de rule_ptr em diante)
        começando na 'seq_sets' (na posição seq_ptr).
        """
        if rule_ptr == len(rule_skeleton):
            return True
        if seq_ptr == len(seq_sets):
            return False
        rule_item = rule_skeleton[rule_ptr]
        
        if rule_item == "GAP":
            for i in range(seq_ptr, len(seq_sets)):
                if self._match_recursive(rule_ptr + 1, i, rule_skeleton, seq_sets):
                    return True 
            return False 
        
        else:
            seq_item = seq_sets[seq_ptr]
            
            if rule_item.issubset(seq_item):
                return self._match_recursive(rule_ptr + 1, seq_ptr + 1, rule_skeleton, seq_sets)
            else:
                return False

    def _check_sequence_match(self, rule_skeleton, seq_sets) -> bool:
        """
        Função "wrapper" que tenta iniciar a correspondência recursiva
        em cada ponto de partida possível na sequência.
        """
        if not rule_skeleton:
            return True 
            
        self._match_recursive.cache_clear() 
        
        for start_idx in range(len(seq_sets)):
            if self._match_recursive(0, start_idx, 
                                     tuple(rule_skeleton), 
                                     tuple(seq_sets)):     
                return True
                
        return False

    def _validate(self, data_string: str, rule_string: str) -> dict:
        """
        Processa o bloco de dados e a regra, calculando o COUNT e a ROC_AUC.
        """
        print(f"Validando regra: {rule_string}")
        
        try:
            rule_skeleton = self._parse_rule_to_skeleton(rule_string)
            if not rule_skeleton:
                return {'count': 0, 'roc_auc': 0.0, 'mean_confidence': 0.0}
        except Exception as e:
            print(f"Erro ao parsear a regra: {e}")
            return {'count': 0, 'roc_auc': 0.0, 'mean_confidence': 0.0}
            
        print(f"Esqueleto da Regra (com GAPs): {rule_skeleton}")

        matching_scores = []
        matching_targets = [] 
        
        lines = data_string.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('sequence,y_true,confidence'):
                continue
                
            try:
                parts = line.rsplit(',', 2)
                seq_str = parts[0]
                target = int(parts[1])  
                confidence = float(parts[2])
            except (IndexError, ValueError):
                print(f"Skipping malformed line: {line[:50]}...")
                continue
            
            seq_sets = self._parse_kosarak_to_sets(seq_str)
            
            if self._check_sequence_match(rule_skeleton, seq_sets):
                matching_scores.append(confidence)
                matching_targets.append(target) 
                
        count = len(matching_scores)
        final_auc = 0.0
        
        if count > 0:
            if len(np.unique(matching_targets)) < 2:
                print(f"Aviso: O padrão '{rule_string}' só foi encontrado em uma classe. AUC é indefinida (retornando 0.0 ou 1.0).")
                final_auc = np.mean(matching_targets) 
            else:
                final_auc = roc_auc_score(matching_targets, matching_scores)
        
        return {
            'count': count,
            'roc_auc': float(final_auc), # <-- NOVO: O valor que você realmente quer
            'mean_confidence': float(np.mean(matching_scores) if count > 0 else 0.0) # O valor antigo
        }

    def validate(self, df, rules):
        for rule in rules:
            rule_string = rule['element']
            validation_result = self._validate(
                data_string="\n".join([f"{row['sequence']},{row['y_true']},{row['confidence']}" for _, row in df.iterrows()]),
                rule_string=rule_string
            )
            print(f"Validação para regra '{rule_string}': {validation_result}")

    