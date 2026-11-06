from datetime import date, datetime, timezone

from zerodte.flux.models import OptionRow, ChainSnapshot, RatioPoint, ScanRow


def test_optionrow_and_chain():
    r = OptionRow(strike=7000.0, is_call=True, bid=40.0, ask=41.0, iv=0.18, delta=0.52)
    c = ChainSnapshot(underlying="SPX", expiry=date(2026, 6, 12),
                      as_of=datetime(2026, 6, 8, 20, 0, tzinfo=timezone.utc), rows=(r,))
    assert c.rows[0].iv == 0.18 and c.expiry == date(2026, 6, 12)


def test_ratiopoint_and_scanrow_fields():
    rp = RatioPoint(as_of=datetime(2026, 6, 8, tzinfo=timezone.utc),
                    front_expiry=date(2026, 6, 12), back_expiry=date(2026, 6, 15),
                    ref_delta=0.35, front_iv=0.22, back_iv=0.20, ratio=1.1)
    assert rp.ratio == 1.1
    sr = ScanRow(front_expiry=date(2026, 6, 12), back_expiry=date(2026, 6, 15), gap_td=1,
                 ratio=1.1, ratio_prev=1.15, drop=0.05, zscore=1.3, flag=True)
    assert sr.drop == 0.05 and sr.flag is True
