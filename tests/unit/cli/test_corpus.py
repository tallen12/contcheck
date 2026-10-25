from contcheck.cli.corpus import CorpusSource, build_corpus_app


def test_corpus_app_has_fetch() -> None:
    app = build_corpus_app()

    assert "fetch" in app
    assert [s.value for s in CorpusSource] == ["wikipedia"]
