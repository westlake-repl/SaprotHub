from typing import Tuple

FOLDSEEK_STRUCT_VOCAB = set("pynwrqhgdlvtmfsaeikc#")


def _looks_like_sa_sequence(seq: str) -> bool:
    if not isinstance(seq, str):
        return False

    upper = lower = 0
    for ch in seq:
        if ch.isupper():
            upper += 1
        elif ch in FOLDSEEK_STRUCT_VOCAB or ch.islower():
            lower += 1

    if upper == 0 or lower == 0:
        return False

    ratio = lower / max(upper, 1)
    return 0.5 <= ratio <= 1.5


def normalize_to_amino_acids(seq: str) -> Tuple[str, bool]:
    """
    Convert a structure-aware (SA) sequence to a plain amino-acid sequence by removing
    the interleaved structural tokens. If the sequence does not look like an SA sequence,
    it is returned unchanged.

    Returns:
        normalized_seq: The processed sequence.
        converted: Whether any conversion happened.
    """
    if not isinstance(seq, str) or not seq:
        return seq, False

    if not _looks_like_sa_sequence(seq):
        return seq, False

    aa_chars = [ch for ch in seq if ch.isupper()]
    if aa_chars:
        return "".join(aa_chars), True

    # Fallback: if no uppercase letters were found (e.g., lowercase AA input),
    # return uppercased sequence.
    return seq.upper(), True

