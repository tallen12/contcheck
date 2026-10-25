from contcheck.corpus.ids import TEXT_ID_LENGTH, text_id


def test_text_id_is_stable_short_and_tells_texts_apart() -> None:
    assert text_id("A story.") == text_id("A story.")
    assert text_id("A story.") != text_id("A story. Again.")
    assert len(text_id("A story.")) == TEXT_ID_LENGTH
