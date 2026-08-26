# Scope and limitations

## Intended demonstration

I use Freddie Mac mortgage data to demonstrate monthly credit-state migration, lifetime
default risk, loss severity, exposure, staging, discounting, scenario weighting, and
provision backtesting. The result is an IFRS9-style research engine, not a production
interpretation of every IFRS 9 requirement.

## Portfolio mismatch

Freddie Mac loans are secured, amortizing mortgages. PD term structures, staging, and
discounted cash shortfalls transfer conceptually to consumer credit, but mortgage LGD levels
and recovery timing do not represent unsecured cards. Mortgage EAD also does not require a
revolving credit conversion factor.

## Bounded and weighted sample

The source population is restricted to 2005, 2006, and 2007 originations. I sample 10,000
loans from each of the twelve quarters and expand aggregates by the source-to-sample ratio.
This balances vintage coverage and memory use, but sampling variance remains. The crisis-era
vintages are also not representative of every underwriting or macroeconomic regime.

## Calendar design and survivor selection

Development ends in December 2011, out-of-time validation covers 2012, and the ECL snapshot
is December 2012. This prevents future labels from entering reporting-date features. It does
not eliminate survivor selection: the snapshot contains only loans still present several
years after origination and after the most severe housing-crisis period.

## Model calibration

The default hazard model has strong OOT rank ordering but underpredicts the weighted event
rate. The cure model materially underpredicts the OOT cure rate, and the severity model
overpredicts mean OOT severity. Convergence is necessary but does not establish calibration.

The portfolio backtest is close in aggregate because Stage 3 overprovisioning offsets large
Stage 1 and Stage 2 shortfalls. Aggregate coverage must therefore be interpreted together
with stage-level results.

## Staging choice

The SICR rule combines relative and absolute increases in modeled remaining-lifetime PD with
a 30 DPD backstop. The thresholds are transparent research choices, not universal regulatory
parameters. At the selected snapshot, quantitative PD triggers add no performing loans to
Stage 2 beyond the delinquency and default backstops. The source also lacks many qualitative
SICR indicators available to a lender.

## Censoring and competing exits

Prepayment, defect removal, and reperforming-loan exits remove a loan from the observed risk
set. Treating these as right-censoring is transparent but may still create informative
censoring. The pipeline preserves each exit type for audit.

## Scenario assumptions

The three scenarios shift monthly hazard odds by fixed multipliers. Their weights and shifts
are judgmental sensitivity assumptions, not estimates from an external macroeconomic
forecast. A production model would link segment PDs and LGDs to governed economic variables,
forecast paths, scenario probabilities, and management overlays.

## ECL and backtest simplifications

Stage 3 uses immediate EAD times LGD, while production measurement would usually model
expected cash shortfalls and recoveries over time. The backtest freezes one reporting date
rather than assessing repeated accounting dates.

The primary loss window is aligned with ECL horizon by stage. A second all-years diagnostic
is retained as a stress view only. It is not a like-for-like accuracy test for Stage 1 because
Stage 1 ECL covers 12 months.

## Reporting convention

The long performance histories span Freddie Mac's May 2019 accounting-cycle change. The
pipeline uses the Release 47 layout and records the pre-May 2019 period flag, but a production
implementation would perform additional governance over convention changes and restatements.
