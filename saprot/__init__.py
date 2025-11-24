import sys
import os

current_file = os.path.abspath(__file__)
saprot_dir = os.path.dirname(current_file)
sys.path.append(saprot_dir)

try:
    from .utils.hf_tokenizer_fallback import (
        patch_auto_tokenizer_with_snapshot_fallback,
    )

    patch_auto_tokenizer_with_snapshot_fallback()
except Exception:
    # The transformers stack might not be available in every environment
    # where saprot is imported (e.g. during lightweight tooling). In that
    # case we simply skip the fallback patching and let downstream callers
    # handle tokenizer loading as usual.
    pass