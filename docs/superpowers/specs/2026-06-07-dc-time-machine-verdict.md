# DC Time Machine — backtest verdict

Cache(s): data/dc_cache_full
Sessions: 84  [2026-01-15 .. 2026-05-15], front_dte=9

Each cell: n trades · tr%=transform rate · maxP%=max-profit hit (of settled transforms) · win% · PF · NET $ total. k=0 mid → k=1 full spread cross.

## All sessions — delta × fill_k
```
             k=0.00                          k=0.50                          k=1.00                        
delta=0.30  n= 74 tr%=  9 maxP%= 43 win%= 54 PF= 1.28 NET=$     990
             n= 74 tr%=  0 maxP%=  0 win%=  0 PF= 0.00 NET=$  -28460
             n= 74 tr%=  0 maxP%=  0 win%=  0 PF= 0.00 NET=$  -25100

delta=0.35  n= 72 tr%= 17 maxP%= 25 win%= 57 PF= 1.46 NET=$    1380
             n= 72 tr%=  0 maxP%=  0 win%=  1 PF= 0.00 NET=$  -28582
             n= 72 tr%=  0 maxP%=  0 win%=  0 PF= 0.00 NET=$  -25710

delta=0.40  n= 79 tr%= 25 maxP%= 32 win%= 59 PF= 3.47 NET=$    5395
             n= 79 tr%=  0 maxP%=  0 win%=  0 PF= 0.00 NET=$  -33260
             n= 79 tr%=  0 maxP%=  0 win%=  0 PF= 0.00 NET=$  -26970

delta=0.45  n= 82 tr%= 54 maxP%= 13 win%= 59 PF= 3.57 NET=$    6485
             n= 82 tr%=  0 maxP%=  0 win%=  0 PF= 0.00 NET=$  -34512
             n= 82 tr%=  0 maxP%=  0 win%=  0 PF= 0.00 NET=$  -31130

```
## His-numbers cross-check — Feb–Apr 2026 (62 sessions)
Author published (3 months): ~106 trades, ~63% win, ~25% max-profit-hit.

```
             k=0.00                          k=0.50                          k=1.00                        
delta=0.30  n= 53 tr%= 13 maxP%= 43 win%= 53 PF= 1.54 NET=$    1455
             n= 53 tr%=  0 maxP%=  0 win%=  0 PF= 0.00 NET=$  -23960
             n= 53 tr%=  0 maxP%=  0 win%=  0 PF= 0.00 NET=$  -18720

delta=0.35  n= 53 tr%= 19 maxP%= 30 win%= 53 PF= 1.39 NET=$    1070
             n= 53 tr%=  0 maxP%=  0 win%=  2 PF= 0.00 NET=$  -24205
             n= 53 tr%=  0 maxP%=  0 win%=  0 PF= 0.00 NET=$  -19480

delta=0.40  n= 58 tr%= 29 maxP%= 35 win%= 59 PF= 4.67 NET=$    4860
             n= 58 tr%=  0 maxP%=  0 win%=  0 PF= 0.00 NET=$  -24775
             n= 58 tr%=  0 maxP%=  0 win%=  0 PF= 0.00 NET=$  -20000

delta=0.45  n= 60 tr%= 55 maxP%= 15 win%= 55 PF= 2.53 NET=$    3340
             n= 60 tr%=  0 maxP%=  0 win%=  0 PF= 0.00 NET=$  -26430
             n= 60 tr%=  0 maxP%=  0 win%=  0 PF= 0.00 NET=$  -25720

```
