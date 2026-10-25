from contcheck.analysis.lemminflect_inflector import LemmInflector


def test_inflects_irregular_and_regular_verbs() -> None:
    inflector = LemmInflector()

    assert inflector.inflect("go", "VBD") == ("went",)
    assert inflector.inflect("walk", "VBZ") == ("walks",)
    assert inflector.inflect("be", "VBD") == ("was", "were")


def test_unknown_tag_gives_no_forms() -> None:
    assert LemmInflector().inflect("go", "XYZ") == ()
