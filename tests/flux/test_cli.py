from zerodte.flux.cli import build_parser


def test_cli_has_serve_and_scan():
    p = build_parser()
    ns = p.parse_args(["serve", "--port", "8787"])
    assert ns.cmd == "serve" and ns.port == 8787
    ns2 = p.parse_args(["scan", "--on-date", "2026-06-10", "--by", "drop"])
    assert ns2.cmd == "scan" and ns2.on_date == "2026-06-10" and ns2.by == "drop"
