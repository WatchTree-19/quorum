"""Tests for the sovereign ratings adapter."""

import csv

from quorum.adapters.sovereign_ratings import is_speculative, load_items, notch


def test_ig_boundary():
    assert is_speculative("BBB-", "SP") == 0
    assert is_speculative("BB+", "SP") == 1
    assert is_speculative("Baa3", "MOODYS") == 0
    assert is_speculative("Ba1", "MOODYS") == 1
    assert is_speculative("BBB-", "FITCH") == 0


def test_default_tier_is_speculative():
    assert is_speculative("SD", "SP") == 1
    assert is_speculative("RD", "FITCH") == 1
    assert is_speculative("C", "MOODYS") == 1


def test_unrated_is_none():
    assert is_speculative("", "SP") is None
    assert is_speculative(None, "FITCH") is None
    assert notch("NOTARATING", "SP") is None


def test_load_items_admits_two_raters_refuses_one(tmp_path):
    p = tmp_path / "panel.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["country", "iso3", "sp", "moodys", "fitch",
                    "gdp_pc_usd", "inflation_pct", "current_account_pct_gdp"])
        w.writerow(["Twoland", "TWO", "BBB-", "", "BB+", "10000", "3.0", "-1.0"])
        w.writerow(["Oneland", "ONE", "AAA", "", "", "50000", "2.0", "1.0"])
    items = load_items(p)
    assert [it.item_id for it in items] == ["TWO"]
    # One IG vote, one speculative vote, two raters: an even tie.
    assert items[0].agreement == "split"
    assert items[0].majority_label is None


def test_majority_and_unanimous(tmp_path):
    p = tmp_path / "panel.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["country", "iso3", "sp", "moodys", "fitch",
                    "gdp_pc_usd", "inflation_pct", "current_account_pct_gdp"])
        # Hungary-shaped: IG, speculative, IG -> majority IG (label 0).
        w.writerow(["Majorityland", "MAJ", "BBB-", "Ba1", "BBB", "20000", "4.0", "1.0"])
        w.writerow(["Safeland", "SAF", "AAA", "Aaa", "AAA", "80000", "2.0", "5.0"])
    items = {it.item_id: it for it in load_items(p)}
    assert items["MAJ"].agreement == "majority"
    assert items["MAJ"].majority_label == 0
    assert items["SAF"].agreement == "unanimous"
    assert items["SAF"].majority_label == 0


def test_rater_affinity_contested_only():
    from quorum.schema import Item, Prediction
    from quorum.affinity import rater_affinity

    items = [
        # Unanimous: must not count towards affinity.
        Item("u1", "t", {}, (0, 0, 0), ("A", "B", "C")),
        # Contested: A and C say 0, B says 1; model says 1 -> agrees with B only.
        Item("c1", "t", {}, (0, 1, 0), ("A", "B", "C")),
        # Contested with an abstention: excluded entirely.
        Item("c2", "t", {}, (1, 0, 1), ("A", "B", "C")),
    ]
    preds = [
        Prediction("u1", 0),
        Prediction("c1", 1),
        Prediction("c2", None),
    ]
    aff = {a.rater_id: a for a in rater_affinity(items, preds)}
    assert aff["A"].n == 1 and aff["A"].agree == 0
    assert aff["B"].n == 1 and aff["B"].agree == 1
    assert aff["C"].n == 1 and aff["C"].agree == 0


def test_history_forward_fill(tmp_path):
    from quorum.adapters.sovereign_history import load_items as load_history

    (tmp_path / "XXX.txt").write_text(
        "S&P | 2020-02-10 | BBB-\n"
        "S&P | 2021-08-01 | BB+\n"
        "Moody's | 2020-01-05 | Baa3\n",
        encoding="utf-8",
    )
    items = load_history(tmp_path, start="2020-01-01", end="2021-12-31")
    by_q = {it.metadata["quarter"]: it for it in items}
    # 2020Q1: both rated IG -> unanimous 0.
    assert by_q["2020Q1"].labels == (0, 0, None)
    # 2021Q2: S&P still BBB- (forward-filled), Moody's Baa3 -> unanimous.
    assert by_q["2021Q2"].labels == (0, 0, None)
    # 2021Q3 on: S&P downgraded to BB+ -> split (two raters, one each way).
    assert by_q["2021Q3"].labels == (1, 0, None)
    assert by_q["2021Q3"].agreement == "split"
    # Quarters before any second rating never become items.
    assert "2019Q4" not in by_q
