"""PIC identities bind source tokens to Writer semantics, excluding public IDs."""

import hashlib
import pickle


def fingerprint(value):
    return hashlib.sha256(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).digest()


def compiled_identity(method_id, input_ids, soft_hash):
    return fingerprint((method_id, input_ids, soft_hash)).hex()
