# Inventory Guard Calibration — Sweep 5e Finding

Date: 2026-05-19
Sweep: `scripts/results/coarse/guards/5e_inv_limits_*/`
Window: ADAUSDC 2024-01-01 → 2024-12-31

## TL;DR

Tightening the inventory soft/hard skew limits **further than the default** does
not improve performance — it strictly hurts. The mid-tightness setting wins,
the very-tight setting is the worst of the three. This contradicts my earlier
"Priority B: tighter guards" recommendation. The inventory guard's job is
to *shape quoting near the limit*, not *prevent the limit from being hit* —
max inventory skew sits at 0.91–0.98 regardless of soft/hard setting because
external price moves dominate.

## Raw results

Same baseline config (post-fix), only the inventory soft/hard limits varied:

| Soft / Hard | ret%   | excess vs hold | Sharpe | DD%  | max_abs_inv_skew |
|-------------|--------|----------------|--------|------|------------------|
| 0.30 / 0.55 | **−9.73**  | −28.05       | 0.17 | 55.1 | 0.915 |
| 0.45 / 0.70 | **+34.94** | +16.62       | 0.80 | 41.0 | 0.984 |
| 0.60 / 0.85 | **+22.86** | +4.54        | 0.65 | 42.5 | 0.952 |

(Default before this exercise was `soft=0.60, hard=0.80`.)

## Why tighter hurts — proposed mechanism

The inventory guard fires soft when `|skew| ≥ soft_limit` and hard when
`|skew| ≥ hard_limit`. When fired it (a) widens the accumulating-side spread
by `soft_spread_mult` or `hard_spread_mult`, and (b) shrinks the accumulating-
side size by `soft_size_mult` or — if `inventory_hard_disable_accumulation_side
=True` — disables that side entirely.

ADA's price-path properties make this delicate:

1. **External price moves dominate the skew trajectory.** Over a year of
   ADA/USDC, the bot reaches `max_abs_inv_skew ≥ 0.91` regardless of soft/hard
   setting. The guard cannot prevent skew from reaching dangerous levels —
   ADA's drift is too large relative to fee-driven re-balancing.

2. **A tight soft limit fires the guard during normal operation, not extremes.**
   With `soft=0.30`, the guard activates on routine inventory imbalance well
   below the actual danger zone. The widening + size-cut on the accumulating
   side reduces fill rate during ordinary chop — *exactly when fee capture is
   highest*. Net effect: kill fee revenue without preventing the bad-tail
   skew incidents that the hard guard exists for.

3. **The hard guard provides the real protection — and it does fire.** Even
   with the looser `hard=0.85`, max skew touches 0.95, meaning the bot is
   pressing against the hard limit regularly. The `disable_accumulation_side
   =True` short-circuit (see [[walkforward_findings]] sweep 5h) is what
   actually keeps inventory from going past 1.0; tightening the threshold
   just trips that mechanism earlier, with marginal benefit.

4. **The "shape, don't prevent" principle.** Think of soft_spread_mult as
   reducing the *probability* of an additional accumulating fill rather than
   blocking it. At skew 0.30 the bot still has 70% headroom — widening here
   just costs fees. At skew 0.60 the bot has 40% headroom and the widening is
   appropriate insurance. At skew 0.80 the hard disable kicks in and stops
   any further accumulation regardless of multiplier.

## What this means for tuning

| Setting             | Effect                                  | Recommendation |
|---------------------|-----------------------------------------|----------------|
| `soft_limit ≪ 0.45` | Fires during normal chop → fee loss     | **Avoid**       |
| `soft_limit 0.45–0.60` | Sweet spot — fires only on real skew | **Use 0.45**    |
| `soft_limit ≥ 0.70` | Fires too late — adverse-selection bleed before guard activates | Default-safe but suboptimal |
| `hard_limit`        | Mostly cosmetic given disable_accum=True | Keep close to soft+0.20-0.25 |
| `inventory_hard_disable_accumulation_side` | Critical — must stay True | **Do NOT disable** |

**Recommended values:** `soft=0.45, hard=0.70` (vs current defaults 0.60/0.80).

## Caveats

- These results are from a single window (ADA/USDC 2024). The 2024 chop
  profile may favour this specific tightness. Walk-forward across 2020–2025
  is needed before treating this as a robust setting.
- The +34.94% absolute return is below the baseline +46.6% seen in the
  ablation, because the guards script's `COMMON_FLAT` uses
  `outer_recenter_trigger_pct=0.65` whereas the ablation used 0.50. The
  *relative* ordering across the three 5e configs is the load-bearing
  result; absolute levels would shift if rerun with the ablation's trigger.
- Sweep 5d (`toxicity_hard_size_mult`) showed a similar pattern:
  `0.6` wins (+88%), `0.0` loses (+11%) — "shape, don't prevent" applies to
  the toxicity guard too.

## Related findings from the same rerun

| Param                          | Default | **Winner** | Δ return |
|--------------------------------|---------|------------|----------|
| `toxicity_buy_ratio_hard`      | 0.80    | **0.85**   | +44pp    |
| `toxicity_hard_size_mult`      | 0.40    | **0.60**   | +41pp    |
| `inventory_skew_soft_limit`    | 0.60    | **0.45**   | +12pp    |
| `inventory_skew_hard_limit`    | 0.80    | **0.70**   | (paired with soft) |

Each of these is a "loosen the response" change: trip the guard later,
shrink size less aggressively when tripped. The corrected interpretation
is that the previous defaults were over-tuned for **avoiding** skew rather
than for **earning fees in normal conditions**. For an ADA-pair LP, fee
capture is the scarcer resource and over-protective guards starve it.

## Open questions

1. **Does the 0.45/0.70 winner generalise out-of-sample?** Run on 2020/21/22/
   25q1 with otherwise-baseline config to test.
2. **Do the four guard wins stack cleanly?** Per-axis winners did NOT stack
   when applied across the full strategy (`BEST_TUNED` walk-forward failed
   on every window), but those were across orthogonal axes. The four guard
   wins are within one subsystem — interaction risk lower but not zero.
   Verify with a joint test before adopting.
3. **Is there a tighter soft that still works if `soft_spread_mult` is also
   lowered?** Sweep 5g showed `soft_spread_mult=1.0` (no widening) gives
   +28.5% vs default 1.3's +46.6% — so loosening the spread multiplier
   without changing the threshold *hurts*. The combination of low-threshold
   + low-mult was not tested.
