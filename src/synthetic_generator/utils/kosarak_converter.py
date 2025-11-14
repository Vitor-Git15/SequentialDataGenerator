def to_kosarak_format(class_label, sequence_data):
    """
    Convert a sequence of itemsets into a Kosarak-like (SPMF) formatted string.
    """
    parts = [str(class_label)]
    for itemset_str in sequence_data:
        parts.append(itemset_str)
        parts.append("-1")
    parts.append("-2")
    return " ".join(parts)