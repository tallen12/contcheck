from contcheck.corpus.hub import HubLoader, HubQuery


def test_query_defaults_to_the_train_split_and_a_large_buffer() -> None:
    query = HubQuery(dataset="org/name")

    assert (query.config, query.data_files, query.data_dir) == (None, None, None)
    assert query.split == "train"
    assert query.shuffle_buffer == 10_000


def test_a_loader_is_any_callable_from_query_and_seed_to_rows() -> None:
    calls: list[tuple[HubQuery, int]] = []

    def load(query: HubQuery, seed: int) -> list[object]:
        calls.append((query, seed))
        return [{"id": 1}]

    loader: HubLoader = load
    query = HubQuery(dataset="org/name", data_dir="data/x")

    assert list(loader(query, 3)) == [{"id": 1}]
    assert calls == [(query, 3)]
