from src_v2.backtest.canonical_ic import price_ic


def test_price_ic_credit_and_settlement():
    chain = {
        (5120.0, True): (4.0, 4.4), (5180.0, True): (1.0, 1.2),    # short/long call
        (4880.0, False): (4.0, 4.4), (4820.0, False): (1.0, 1.2),  # short/long put
    }
    r = price_ic(chain, spot=5000.0, em=120.0, em_mult=1.0, wing=60,
                 step=5, slippage=0.10, settle_spot=5000.0)
    # credit = (4.2-1.1-0.10)*2 sides = 3.0*2 = 6.0 ; settle ATM -> no loss
    assert abs(r["credit"] - 6.0) < 1e-6
    # pnl = 6.0*100 - 4 commission
    assert abs(r["pnl"] - (6.0 * 100 - 4)) < 1e-6


def test_price_ic_loss_capped_at_wing():
    chain = {
        (5120.0, True): (4.0, 4.4), (5180.0, True): (1.0, 1.2),
        (4880.0, False): (4.0, 4.4), (4820.0, False): (1.0, 1.2),
    }
    r = price_ic(chain, spot=5000.0, em=120.0, em_mult=1.0, wing=60,
                 step=5, slippage=0.10, settle_spot=5400.0)  # blows through call wing
    assert abs(r["call_loss"] - 60) < 1e-6   # capped at wing
