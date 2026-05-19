# CL-AMM Walk-Forward Findings (ADAUSDC, 2020–2025q1)

Generated 2026-05-18 from the coarse-sweep + ablation + walk-forward exercise.

## TL;DR

The current CL-AMM strategy does **not generalise across 5 years of ADA/USDC data**.
The +46.6% return on 2024 is regime-specific; the same config loses to buy-and-hold
in **every** other window (2020, 2021, 2022, 2023, 2025q1). Combining per-axis
sweep "winners" into a single `BEST_TUNED` config makes things strictly worse —
even 2024 drops to −21%. **Per-axis winners do not stack.**

Without a hedge (a hard product constraint), ADA's volatility creates LVR that
swamps fee capture in most years. Three forward paths: pivot to a lower-vol
pair, re-optimise for robustness rather than absolute return, or accept narrow
viability and gate the bot to specific market conditions.

---

## Bugs fixed in this round

1. **Concentration silent clamp** (`backtest_strategies.py:246-260`,
   `deltadefi_cl_amm_mm.py:330-340`) — base_concentration_pct outside
   `[min_concentration, max_concentration]` was silently clamped at the
   corridor boundary. Past sweeps over `concentration=3` with default
   `min=5` actually ran at 5. Now raises ValueError.
2. **`outer_spread_pct_of_range` wiring gap** (`backtest_strategies.py:564-574`)
   — param defined in live strategy but not read by backtest. Live placed
   outer orders at `max(inner_spread × outer_spread_mult, concentration × outer_spread_pct_of_range)`;
   backtest only used the first term. At conc=15 / base_spread=50bps the live
   outer sat at ±9% from mid, backtest at ±1.25% — 7× narrower.
3. **Boolean parse bug** (`backtest_sweep.py:117-135`) — `_parse_value("False")`
   returned the truthy string `"False"`. Sweeps over boolean params silently
   ran every cell as `True`. Now returns `False` correctly.
4. **Shell-script double `--set`** (`run_coarse_guards.sh`, `run_coarse_trend.sh`)
   — argparse with `nargs="*"` keeps only the LAST `--set` flag, so chained
   `--set X=A --set Y=B` dropped X=A entirely. Affected sweeps 5b, 5e, 7a, 7b.

## Ablation under the fixed wiring (2024, full year)

| Config       | ret%   | hold% | excess | sharpe | DD%  | fills  | fee% | lvr% |
|--------------|--------|-------|--------|--------|------|--------|------|------|
| 1_dumb_lp    | −6.4   | +18.3 | −24.8  | 0.20   | 55.9 | 20697  | 52.1 | 93.9 |
| 2_plus_inv   | −7.6   | +18.3 | −26.0  | 0.12   | 52.4 | 16443  | 41.9 | 81.6 |
| 3_plus_tox   | −3.0   | +18.3 | −21.3  | 0.26   | 54.4 | 18915  | 45.6 | 82.3 |
| 4_plus_asym  | −7.0   | +18.3 | −25.3  | 0.19   | 56.2 | 20444  | 51.7 | 94.0 |
| 5_plus_dyn   | +6.0   | +18.3 | −12.4  | 0.41   | 47.0 | 20362  | 53.8 | 83.8 |
| **6_all_on** | **+46.6** | +18.3 | **+28.3** | **0.94** | 47.4 | 14259  | 33.0 | 59.7 |

The full-stack `all_on` config beats every partial-stack permutation by 40+ pp.
**Features interact superlinearly** — none of them alone get within 40pp of
the combined return. This is what motivated the BEST_TUNED experiment.

## ALL_ON vs BEST_TUNED — what changed

These are the 11 parameters BEST_TUNED moved off ALL_ON. Each was selected
because it was the winning value in a coarse one-at-a-time sweep run with
all other params at ALL_ON defaults.

| Param                          | ALL_ON | BEST_TUNED | Source sweep |
|--------------------------------|--------|------------|--------------|
| `max_concentration`            | 40     | **20**     | 3d (max conc) |
| `min_spread_bps`               | 20     | **40**     | 6b (min spread) |
| `natr_range_scale`             | 1.0    | **0.5**    | 3a (NATR scale) |
| `outer_capital_fraction`       | 0.30   | **0.40**   | 4a (outer cap frac) |
| `outer_recenter_trigger_pct`   | 0.50   | **0.65**   | (defensive default) |
| `outer_spread_mult`            | 2.5    | **3.5**    | 4d (outer spread mult) |
| `skew_sensitivity`             | 0.5    | **2.0**    | 6a (skew sens) |
| `toxicity_buy_ratio_hard`      | 0.80   | **0.85**   | 5b (tox buy ratio) |
| `toxicity_hard_size_mult`      | 0.40   | **0.60**   | 5d (tox hard size) |
| `toxicity_sell_ratio_hard`     | 0.80   | **0.85**   | 5b (mirror) |
| `trend_sensitivity`            | 0.5    | **0.0**    | trend layer off |

Each looked like a clean win in its own sweep. Together they break the model.

## Walk-forward results

`scripts/run_best_tuned.py --preset {all_on, best_tuned}` walks each preset
across yearly windows.

### ALL_ON walk-forward

| Window           | ret%   | hold%  | excess  | sharpe | DD%  | beats hold? |
|------------------|--------|--------|---------|--------|------|-------------|
| 2020 (bull)      | −18.2  | +225.0 | −243.2  | 0.13   | 68.7 | ✗           |
| 2021 (bull)      | +176.6 | +331.0 | −154.5  | 1.54   | 64.2 | ✗           |
| 2022 (bear)      | −39.5  | −33.9  | −5.6    | −0.51  | 58.0 | ✗ (worse!)  |
| 2023 (range)     | +75.9  | +91.3  | −15.4   | −8.14  | 32.8 | ✗           |
| **2024 (calm)**  | **+46.6** | +18.3 | **+28.3** | 0.94 | 47.4 | **✓**     |
| 2025q1 (mild −)  | −17.9  | −10.5  | −7.4    | −0.46  | 42.3 | ✗           |

**Gate (5 OOS windows):**
- positive return:        2/5 ❌
- positive excess vs hold: 0/5 ❌
- sharpe > 0.5:           1/5 ❌
- max DD ≥ 60% (bad):     2/5 ❌

**Verdict: FAIL — config overfit to 2024.**

### BEST_TUNED walk-forward

| Window           | ret%   | hold%  | excess  | sharpe | DD%  |
|------------------|--------|--------|---------|--------|------|
| 2020             | −41.8  | +225.0 | −266.8  | −0.35  | 66.2 |
| 2021             | +152.7 | +331.0 | −178.3  | 1.46   | 55.4 |
| 2022             | −65.6  | −33.9  | −31.6   | −1.63  | **78.3** |
| 2023             | +76.7  | +91.3  | −14.7   | −7.93  | 32.6 |
| **2024**         | **−21.1** | +18.3 | −39.4 | −0.09  | 56.9 |
| 2025q1           | −24.6  | −10.5  | −14.1   | −0.82  | 47.1 |

**BEST_TUNED is strictly worse than ALL_ON on every window**, including the
2024 in-sample where it was supposedly tuned. The per-axis winners interact
destructively when stacked. Likely culprits:

- `skew_sensitivity=2.0` × `min_spread_bps=40`: the floor at 40bps clips
  the ask spread at any inventory skew > 0.10 (when base_spread=50bps).
  The bot quotes the floor on the unloading side almost always → relentless
  one-sided fills → inventory blows out faster than the guards can react.
- `max_concentration=20` × wider outer band: the inner band can only widen
  to 20% before clamping. With a wider outer (`outer_spread_mult=3.5`),
  the band structure becomes unbalanced — inner can't accommodate vol
  spikes, outer catches too much.

## Per-axis observations from coarse sweeps

Caveat: many of these were measured with the broken shell-script `--set`
(masked some inventory results) and the broken outer-band wiring
(undercounted outer-band effects). Rerun pending — the *direction* of each
finding likely holds but *magnitudes* are unreliable until reruns complete.

- **Dynamic concentration (#3):** `natr_range_scale=0.5` won by ~23pp over
  default 1.0 on 2024. `max_concentration=20` won by ~42pp over the widened
  40 — i.e. my own widening to fit the old grid was a regression.
- **Outer dual-range (#4):** `outer_capital_fraction=0.40` won by ~24pp.
  Disabling the outer band (`=0.0`) collapsed returns to +3.93% — the band
  is essential. `outer_spread_pct_of_range` showed no variation only because
  the param wasn't wired (now fixed).
- **Guards (#5):** Toxicity actively contributes. Inventory guard's soft
  limit IS firing in 2024 — sweep `inventory_soft_spread_mult` varied
  returns from +29% to +45%. The hard limit's `hard_size_mult` had no
  effect because `inventory_hard_disable_accumulation_side=True` short-circuits
  it. Inventory ablation confirmed max_abs_inv_skew reaches 0.95 — the
  guard IS needed; do not remove without a replacement mechanism.
- **Asymmetric spread (#6):** U-shaped response in `skew_sensitivity`:
  0.0 (symmetric) +66%, 0.5 (mild) +45%, 2.0 (aggressive) +102%. The
  discontinuity is the `min_spread_bps` floor clipping the ask side once
  `skew × sens` is large enough. NOT a bug — a mechanism interaction.
- **Trend (#7):** Direction-dependent. `trend_sensitivity=1.5` gives +208%
  in bull_2021q1 (beats hold by 61pp). Same sens gives −28% in bear_2022
  (loses to hold by 6pp). Fixed sens=0 is the safer default given that
  phase6 indicator validation found near-zero correlation between trend
  signals and forward returns.

## Bigger picture — the LVR problem

Phase 6 (`results/experiment/phase6_results.json`) measured indicator
predictive power on 2024 candles:

- `corr(eff_trend, fwd_|return|)` ≈ 0.03–0.05 at 15m/1h/4h (no direction prediction)
- `corr(NATR, fwd_vol)` ≈ 0.6–0.74 at 15m/1h/4h (volatility IS predictable)
- Hurst ratios ≈ 1.05–1.11 (signals are barely above noise)

So we can predict volatility but not direction. A market-maker on a
volatile asset gets selected against in the predictable-direction case
(adverse fills as price runs); the only edge is in capturing fees during
**range-bound chop**. 2024 had more chop than most years; the strategy
worked. Years where price runs sharply (2020/2021 up, 2022 down) deliver
LVR that swamps fees.

Without a hedge, this is structural — you can't tune your way out of LVR
when inventory drifts the wrong way and you can't unwind on a perp.
The user's hard "no hedge" constraint bounds the achievable Sharpe.

## Forward paths

1. **Pivot the pair** — ETH/USDC or BTC/USDC have lower vol-per-fees than
   ADA. Same config on those pairs may walk-forward. Cheapest experiment.
2. **Reframe the objective** — optimize for `min(yearly_return)` or
   `5th-percentile`, not raw return on 2024. Will reject regime-narrow
   configs. Higher chance of producing a deployable strategy.
3. **Accept narrow viability** — gate the bot to specific NATR / momentum
   regimes, sit out the rest. Real product but lower utilization.

Path 1 first (fast, decisive); then Path 2 on whatever pair survives.
Path 3 is the fallback if neither generalises.

## Artifacts

- Coarse sweeps:               `scripts/results/coarse/{dyn_concentration,outer,guards,asym,trend}/`
- Ablation (full year 2024):   `scripts/results/ablation/ablation_*.csv`
- Walk-forward CSVs:           `scripts/results/best_tuned/walk_forward_{all_on,best_tuned}_*.csv`
- Reproducer scripts:
  - `scripts/run_ablation.py` — 6-config feature ablation
  - `scripts/run_best_tuned.py` — preset-based walk-forward
  - `scripts/run_coarse_{dyn,outer,guards,asym,trend}.sh` — per-axis coarse sweeps
  - `scripts/run_coarse_all.sh` — master runner
