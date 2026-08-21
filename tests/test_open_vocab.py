"""Open-vocabulary label table (rcdl.LabelMap).

No decode of its own: a YOLOE build is an ordinary LTRB head whose class axis
means words, so the only new runtime state is the class_id -> prompt mapping.
These tests need only the compiled module — the real-model path (does the
vocabulary actually change what the model finds?) is in
tests/test_tasks_board_py.py.

    PYTHONPATH=build:python pytest -s tests/test_open_vocab.py
"""

import pytest

rcdl = pytest.importorskip("rcdl")

pytestmark = pytest.mark.skipif(not hasattr(rcdl, "LabelMap"),
                                reason="compiled rcdl module without LabelMap")


def test_from_list_round_trips():
    lm = rcdl.LabelMap.from_list(["sneakers", "jeans", "hoodie"])
    assert len(lm) == 3
    assert list(lm.names) == ["sneakers", "jeans", "hoodie"]
    assert lm.name(0) == "sneakers"
    assert lm[2] == "hoodie"


def test_out_of_range_names_do_not_throw():
    """Decode may legitimately emit an id past a truncated table while
    debugging; naming it must not turn that into an exception."""
    lm = rcdl.LabelMap.from_list(["a", "b"])
    assert lm.name(-1) == "?"
    assert lm.name(2) == "?"
    with pytest.raises(IndexError):      # __getitem__ is the strict accessor
        lm[2]


def test_from_file_trims_and_drops_blank_lines(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("  sneakers  \n\n\tjeans\r\n\nlicense plate\n\n", encoding="utf-8")
    lm = rcdl.LabelMap.from_file(str(p))
    assert list(lm.names) == ["sneakers", "jeans", "license plate"]


def test_from_file_reports_a_missing_or_empty_file(tmp_path):
    with pytest.raises(Exception):
        rcdl.LabelMap.from_file(str(tmp_path / "nope.txt"))
    empty = tmp_path / "empty.txt"
    empty.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(Exception):
        rcdl.LabelMap.from_file(str(empty))


def test_require_size_catches_a_stale_vocabulary():
    """The failure this guards against is silent by construction: a labels file
    from a DIFFERENT build does not move a single box or change a single score,
    it only renames the results — and dropping one word shifts every name after
    it by one."""
    lm = rcdl.LabelMap.from_list(["person", "bicycle", "car"])
    lm.require_size(3)                       # matches: no throw
    with pytest.raises(Exception):
        lm.require_size(4)
    with pytest.raises(Exception):
        lm.require_size(2)


def test_multi_word_prompts_survive_the_file_round_trip(tmp_path):
    """Prompts are phrases, not identifiers — 'license plate' must not come back
    split, and internal spacing must be preserved."""
    words = ["license plate", "street lamp", "double-decker bus"]
    p = tmp_path / "v.txt"
    p.write_text("\n".join(words) + "\n", encoding="utf-8")
    assert list(rcdl.LabelMap.from_file(str(p)).names) == words
