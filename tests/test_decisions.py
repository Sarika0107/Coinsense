"""Tests for the decision rules (no camera or model needed): python -m pytest tests"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.decisions import (Detection, make_decision, count_coins,
                               total_in_cents, iou, TotalStabilizer)


def coin(name, x, conf=0.9):
    return Detection(name, conf, (x, 0, x + 50, 50))   # separate, non-overlapping boxes


def test_total_adds_coin_values():
    dets = [coin("quarter", 0), coin("dime", 100), coin("nickel", 200), coin("penny", 300)]
    assert total_in_cents(count_coins(dets)) == 41


def test_no_coins():
    d = make_decision([])
    assert d.status == "NO_COINS" and d.message == "No coins detected"


def test_rule1_enough():
    dets = [coin("quarter", i * 100) for i in range(10)]          # $2.50
    assert make_decision(dets, target_cents=250).status == "ENOUGH"


def test_rule1_need_more():
    d = make_decision([coin("quarter", 0)], target_cents=100)
    assert d.status == "NEED_MORE" and "$0.75" in d.message


def test_rule2_roll_ready():
    dets = [coin("quarter", i * 100) for i in range(40)]
    assert any("quarter roll ready" in r for r in make_decision(dets).roll_ready)


def test_rule3_overlap():
    dets = [Detection("dime", 0.9, (0, 0, 50, 50)), Detection("dime", 0.9, (10, 10, 60, 60))]
    d = make_decision(dets)
    assert d.status == "SPREAD_OUT" and not d.is_final


def test_rule3_unsure():
    d = make_decision([coin("penny", 0, conf=0.6)])
    assert d.status == "SPREAD_OUT" and d.unsure_count == 1


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_stabilizer_waits_one_second():
    t = [0.0]
    s = TotalStabilizer(1.0, clock=lambda: t[0])
    assert not s.update(50)
    t[0] = 0.5; assert not s.update(50)
    t[0] = 1.1; assert s.update(50)
    t[0] = 1.2; assert not s.update(75)     # total changed -> not final again
