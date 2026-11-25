import pandas as pd

from sales_anomalies import calibrate, evaluate
from sales_anomalies.detect import Episode
from sales_anomalies.truth import TruthEvent

DAY = pd.Timestamp("2023-05-10")


def fraud(amount=9000.0):
    contribution = pd.DataFrame({"date": [DAY], "category": ["Electronics"], "channel": ["direct"],
                                 "sales": [amount], "margin_sold": [amount / 10], "return_cost": [0.0]})
    return TruthEvent("dg-1", "anomaly_data", "fraud_ring", "single_account", DAY, DAY, ["Electronics|direct"],
                      ["sales", "margin"], contribution=contribution)


def episode(segment="Electronics|direct", seen=10000.0, direction="up", start=DAY, end=DAY, metric="sales"):
    e = Episode(segment, "cell", metric, direction, start, end, 5.0, [1])
    e.observed, e.expected = 20000.0 + seen, 20000.0
    return e


def test_a_fraud_ring_explains_an_alert_it_made_and_not_one_it_did_not():
    assert evaluate.explains(fraud(9000.0), episode(seen=10000.0))
    assert not evaluate.explains(fraud(900.0), episode(seen=10000.0))       # 9% of the jump: something else
    assert evaluate.explains(fraud(9000.0), episode(segment="*|*", seen=10000.0))   # a parent that it carried
    assert not evaluate.explains(fraud(9000.0), episode(segment="Clothing|*", seen=10000.0))


def test_an_outage_explains_the_drop_that_day_and_the_jump_after():
    outage = TruthEvent("dg-2", "anomaly_data", "shop_outage", "outage", DAY, DAY + pd.Timedelta(days=2), ["*|*"],
                        ["sales", "margin", "costs", "conversion"])
    assert evaluate.explains(outage, episode("*|*", -50000.0, "down"))
    assert evaluate.explains(outage, episode("*|*", 20000.0, "up", DAY + pd.Timedelta(days=1),
                                             DAY + pd.Timedelta(days=1)))
    later = DAY + pd.Timedelta(days=1)
    assert not evaluate.explains(outage, episode("*|*", -50000.0, "down", later, later))


def test_calibration_never_goes_below_four():
    assert calibrate.GRID[0] == 4.0
    assert calibrate.BUDGET == 0
