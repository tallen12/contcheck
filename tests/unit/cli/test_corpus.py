from pathlib import Path

import pytest

from contcheck.cli.corpus import CorpusSource, article_source, build_corpus_app, merge, report
from contcheck.corpus.jsonl import read_jsonl, write_jsonl
from contcheck.corpus.record import Record, RecordSource, Split


def test_corpus_app_has_fetch() -> None:
    app = build_corpus_app()

    assert "fetch" in app
    assert [s.value for s in CorpusSource] == [
        "wikipedia",
        "simple-wikipedia",
        "gutenberg",
        "fineweb-edu",
        "wikinews",
        "wikibooks",
        "wikivoyage",
        "stackexchange",
        "tinystories",
        "simplestories",
        "cosmopedia-stories",
    ]


@pytest.mark.parametrize("source", list(CorpusSource))
def test_every_source_builds_without_network_access(source: CorpusSource) -> None:
    articles = article_source(source, seed=1)

    assert callable(articles.articles)


def test_merge_writes_one_file_per_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "rule.jsonl"
    write_jsonl(source, [make_record("d1", Split.TRAIN), make_record("d2", Split.TEST)])

    merge([source], out_dir=tmp_path / "out")

    assert [r.doc_id for r in read_jsonl(tmp_path / "out" / "train.jsonl", Record)] == ["d1"]
    assert [r.doc_id for r in read_jsonl(tmp_path / "out" / "test.jsonl", Record)] == ["d2"]
    assert (tmp_path / "out" / "val.jsonl").read_text(encoding="utf-8") == ""
    assert "wrote 1 records" in capsys.readouterr().out


def test_report_prints_counts_and_targets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "rule.jsonl"
    write_jsonl(source, [make_record("d1", Split.TRAIN)])
    config = tmp_path / "gen.yaml"
    config.write_text(
        "seed: 1\nvariants_per_paragraph: 1\n"
        "errors_per_paragraph: {0: 1}\nkind_weights: {default: 1}\n",
        encoding="utf-8",
    )

    report(source)
    plain = capsys.readouterr().out
    report(source, config=config)
    compared = capsys.readouterr().out

    assert "records: 1" in plain
    assert "100.0%  100.0%" in compared
    # Kinds with a corruptor show against their target even with no edits.
    assert "Spelling" not in plain
    assert "Spelling" in compared


def make_record(doc_id: str, split: Split) -> Record:
    return Record(
        doc_id=doc_id,
        paragraph=0,
        variant=0,
        split=split,
        source=RecordSource.RULE,
        clean="a b",
        noisy="a b",
        words=("a", "b"),
        labels=("O", "O"),
        edits=(),
    )
