"""Votación de dorsales."""

from bib_timing.ocr import BibVotes


def test_majority_wins():
    v = BibVotes()
    for _ in range(5):
        v.add("132", 0.8, 1)
    for _ in range(2):
        v.add("182", 0.9, 2)
    v.add("732", 0.95, 3)
    bib, conf, counts = v.winner()
    assert bib == "132"
    assert counts["132"] == 5
    assert conf == 5 / 8


def test_tie_breaks_on_score():
    v = BibVotes()
    v.add("10", 0.5, 1)
    v.add("20", 0.9, 2)
    bib, _, _ = v.winner()
    assert bib == "20"
