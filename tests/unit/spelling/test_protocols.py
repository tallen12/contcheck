from contcheck.spelling.protocols import Dictionary
from tests.conftest import FakeDictionary


def test_dictionary_contract(dictionary: FakeDictionary) -> None:
    implementation: Dictionary = dictionary

    assert implementation.contains("store")
    assert implementation.suggestions("stor", max_distance=1)[0].term == "store"
    assert implementation.suggestions("zzzzzz", max_distance=1) == []
