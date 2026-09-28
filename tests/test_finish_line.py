"""Tests unitarios sin GPU ni video."""

from bib_timing.finish_line import FinishLine, Point, crossed_line


def test_crossed_line_vertical_motion():
    line = FinishLine(Point(0, 100), Point(200, 100))
    assert crossed_line(Point(50, 80), Point(50, 120), line)
    assert not crossed_line(Point(50, 80), Point(50, 90), line)


def test_crossed_line_outside_bbox():
    line = FinishLine(Point(0, 100), Point(100, 100))
    # Cruza la recta infinita pero lejos del segmento
    assert not crossed_line(Point(500, 80), Point(500, 120), line)
