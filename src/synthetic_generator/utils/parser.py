import re

def validate_parser_input(input_string: str) -> (bool, str):
    """
    Valida a string de entrada com base nas regras de sintaxe para sequências,
    adjacência e itemsets.

    Regras:
    1. Elementos são (A-Z_0-9)+
    2. [] indicam adjacência.
    3. {} indicam itemsets.
    4. Não são permitidos colchetes aninhados (ex: [A [B] C]).
    5. Não são permitidos colchetes dentro de chaves (ex: {A [B] C}).
    6. Não são permitidas chaves ou colchetes vazios.
    7. Não são permitidos parênteses/chaves sem correspondência.
    """
    
    invalid_chars = re.search(r"[^A-Z_0-9\[\]{}\s]", input_string)
    if invalid_chars:
        return False, f"Error: Invalid character detected: '{invalid_chars.group(0)}'"
    
    if re.search(r"\[\s*\]", input_string):
        return False, "Error: Empty brackets '[]' are not allowed."
    if re.search(r"{\s*}", input_string):
        return False, "Error: Empty braces '{}' are not allowed."

    tokens = re.findall(r"\[|\]|{|}|[A-Z_0-9]+", input_string)
    
    if not tokens and input_string.strip():
        return False, "Error: The string does not contain valid tokens."

    bracket_level = 0
    brace_level = 0
    
    for token in tokens:
        if token == '[':
            # Proíbe [ dentro de [
            if bracket_level > 0:
                return False, "Error: Nested brackets (e.g., '[...[...]]') detected."
            bracket_level += 1
            
        elif token == ']':
            if bracket_level == 0:
                return False, "Error: Unmatched closing bracket ']' detected."
            bracket_level -= 1

        elif token == '{':
            brace_level += 1
            
        elif token == '}':
            if brace_level == 0:
                return False, "Error: Unmatched closing brace '}' detected."
            brace_level -= 1

    if bracket_level > 0:
        return False, "Error: Unmatched opening bracket '[' detected."
    if brace_level > 0:
        return False, "Error: Unmatched opening brace '{' detected."

    return True, "Valid input."