def get_clean_vocabulary(vocabulary, signal_rules):
    """
    Remove todos os itens presentes nas regras de sinal do vocabulário principal.
    Lida com itens únicos, sequências ordenadas e itemsets ({A B C}).
    """
    items_to_exclude = set()
    for rule in signal_rules:
        element = rule['element']
        
        if element.startswith('{') and element.endswith('}'):
            # É um itemset: remove as chaves e divide
            items = element.strip('{}').split(' ')
        else:
            # É um item único ou uma sequência ordenada
            items = element.split(' ')
            
        items_to_exclude.update(items)
    
    cleaned_vocabulary = [v for v in vocabulary if v not in items_to_exclude]
    return cleaned_vocabulary