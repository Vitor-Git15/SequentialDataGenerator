import random
import re

class SequenceGenerator:
    def __init__(self, vocabulary, min_itemset=1, max_itemset=1, noise_density=1.0):
        self.vocabulary = vocabulary
        self.min_itemset = min_itemset
        self.max_itemset = max_itemset
        self.noise_density = noise_density
        
    def _tokenize(self, rule_string):
        pattern = r'\[|\]|\{|\}|[a-zA-Z0-9_]+'
        return re.findall(pattern, rule_string)

    def _parse(self, tokens):
        parsed_sequence = []
        inside_gap_group = False
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token == '[':
                inside_gap_group = True
                i += 1
                continue
            elif token == ']':
                inside_gap_group = False
                i += 1
                continue
            
            item_obj = {
                'type': 'item',
                'value': None,
                'inside_gap_group': inside_gap_group
            }

            if token == '{':
                itemset_members = []
                i += 1
                while i < len(tokens) and tokens[i] != '}':
                    itemset_members.append(tokens[i])
                    i += 1
                item_obj['value'] = f"{' '.join(itemset_members)}"
                item_obj['type'] = 'itemset'
            else:
                item_obj['value'] = token
            
            parsed_sequence.append(item_obj)
            i += 1
            
        return parsed_sequence
    
    def _noisify_item(self, base_element):
        return (
            f"{base_element}_s{random.randint(1000, 9999)}"
            if random.random() >= self.noise_density
            else base_element
        )

    def _get_noise(self):
        if self.max_itemset > 1:
            itemset_sizes = range(self.min_itemset, self.max_itemset)
            weights = [1 / (i * i) for i in itemset_sizes]
            noise_item_size = random.choices(itemset_sizes, weights=weights, k=1)[0]
        else:
            noise_item_size = 1

        items = [
            self._noisify_item(item)
            for item in random.choices(self.vocabulary, k=noise_item_size)
        ]

        return f"{{ {' '.join(items)} }}" if noise_item_size > 1 else items[0]

    def _get_noise_tokens(self, quantity):
        """Retorna uma lista com N tokens aleatórios do vocabulário"""
        return [self._get_noise() for _ in range(quantity)]

    def generate(self, rule_string, min_len, max_len):
        tokens = self._tokenize(rule_string)
        parsed_structure = self._parse(tokens)
        
        base_len = len(parsed_structure)

        if base_len >= max_len:
            return [node['value'] for node in parsed_structure]

        actual_min = max(min_len, base_len)
        target_length = random.randint(actual_min, max_len)
        
        total_noise_slots = target_length - base_len
        
        if total_noise_slots <= 0:
             return [node['value'] for node in parsed_structure]

        valid_buckets = []
        
        valid_buckets.append('PREFIX')
        valid_buckets.append('SUFFIX')

        for i in range(len(parsed_structure) - 1):
            current_node = parsed_structure[i]
            next_node = parsed_structure[i+1]
            
            if current_node['inside_gap_group'] and next_node['inside_gap_group']:
                valid_buckets.append(i)

        bucket_counts = {key: 0 for key in valid_buckets}
        
        for _ in range(total_noise_slots):
            chosen_bucket = random.choice(valid_buckets)
            bucket_counts[chosen_bucket] += 1
            
        final_sequence = []
        
        final_sequence.extend(self._get_noise_tokens(bucket_counts.get('PREFIX', 0)))
        
        for i, node in enumerate(parsed_structure):
            final_sequence.append(node['value'])
            if i in bucket_counts:
                final_sequence.extend(self._get_noise_tokens(bucket_counts[i]))
        final_sequence.extend(self._get_noise_tokens(bucket_counts.get('SUFFIX', 0)))
        
        return final_sequence

    def format_output(self, sequence):
        output_str = []
        for item in sequence:
            output_str.append(str(item))
        return " ".join(output_str)
    