from timeentropy.provenance import sha256_json


def test_json_hash_is_key_order_independent() -> None:
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})
