from itertools import islice

import pytest

from contcheck.corpus.hub import HubQuery, stream_query


@pytest.mark.datasets
def test_stream_query_reads_rows_of_a_data_file_pattern() -> None:
    query = HubQuery(dataset="Fumika/Wikinews-multilingual", shuffle_buffer=4)

    rows = list(islice(stream_query(query, 0), 2))

    assert len(rows) == 2
    assert all(isinstance(row, dict) and "text" in row for row in rows)
