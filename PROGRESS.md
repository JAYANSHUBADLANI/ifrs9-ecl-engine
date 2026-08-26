# Project progress

Last updated: 2026-08-20

## Completed

- Audited all twelve supplied Freddie Mac archives without extracting them.
- Bound ingestion to Release 47 with 31 origination and 35 performance fields.
- Implemented direct ZIP streaming, strict schema checks, deterministic quarterly sampling,
  safe checkpoints, state mapping, adjacent-month transitions, and Actual Loss reconciliation.
- Built Phase 1 on 120,000 sampled loans and 8,068,208 monthly performance rows.
- Reconciled all 8,487 disclosed Actual Loss rows within one cent.
- Estimated 7,948,131 valid adjacent transitions with no duplicate loan-months or unknown states.
- Fitted and validated the weighted discrete-time lifetime PD model.
- Fitted and validated cure and conditional severity models.
- Implemented componentized reporting EAD and contractual amortizing EAD projections.
- Implemented SICR rules, Stages 1 to 3, monthly discounting, and three weighted scenarios.
- Calculated ECL for 36,962 December 2012 snapshot loans.
- Implemented a horizon-aligned provision backtest and a separate ultimate-loss stress view.
- Generated model checkpoints, aggregate tables, diagnostics, and figures for Phases 0 to 5.
- Kept raw archives and derived loan-level data outside version control.

## Final headline results

- Phase 2 OOT weighted ROC AUC: 0.96685.
- Phase 2 OOT weighted mean prediction: 0.2334% versus 0.2802% event rate.
- Cure OOT ROC AUC: 0.63141.
- Cure OOT predicted rate: 62.73% versus 91.20% observed.
- Severity OOT predicted mean: 55.98% versus 51.47% observed.
- EAD OOT MAE: 160.66 across 462,388 rows.
- Snapshot stages: 31,521 Stage 1, 1,923 Stage 2, and 3,518 Stage 3.
- Expansion-weighted probability-weighted ECL: 6.994 billion.
- Horizon-aligned discounted loss: 7.121 billion.
- Horizon-aligned coverage: 98.22%, with a 126.71 million shortfall.
- Ultimate all-years stress coverage: 82.32%.

## Interpretation flags

- Aggregate backtest closeness is driven by offsets: Stage 1 and Stage 2 underprovide while
  Stage 3 overprovides.
- PD SICR tests add no incremental Stage 2 loans beyond delinquency backstops at this snapshot.
- Cure predictions materially understate the observed OOT cure rate.
- Scenario multipliers are judgmental sensitivities and not fitted macroeconomic forecasts.
- This is a secured mortgage research sample, not a production consumer-card IFRS 9 model.

## Final verification

- All 244 automated tests pass under Python 3.12.
- Source and tests compile successfully.
- Pyflakes reports no unused imports or undefined names.
- The changed Phase 5 files pass Black formatting checks.
- Stage, scenario, provision, and realized-loss aggregate tables reconcile to their summaries.
- The final Phase 4 and Phase 5 figures were visually inspected.
