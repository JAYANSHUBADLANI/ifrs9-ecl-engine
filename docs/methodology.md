# Modeling methodology

## Portfolio and sampling

I use every origination quarter from 2005Q1 through 2007Q4. Within each quarter, I select
10,000 loans by the smallest stable loan-ID hashes. This gives 120,000 sampled loans and
prevents a large quarter from crowding out a smaller one. Every aggregate estimate carries
the quarter population divided by 10,000 as its expansion weight.

This is a stratified research sample, not the full 4,043,191-loan population. The design
keeps complete monthly histories for selected loans, including records through March 2026.
It is reproducible because the hash seed and the quarterly sample size are committed in the
project configuration.

## Calendar split

- Development target months end in December 2011.
- Out-of-time validation target months run from January through December 2012.
- The ECL reporting snapshot is December 2012.
- Backtest losses occur strictly after December 2012. Stage 1 is evaluated through December
  2013, Stage 2 through each loan's modeled remaining lifetime, and Stage 3 through March 2026.

Features for a one-month hazard target are taken at month t. Default is observed only from
the adjacent record at month t plus one. Later rows are never used to create reporting-date
features, staging inputs, or ECL.

## Credit states and roll rates

The panel groups delinquency into CURRENT, 30_DPD, 60_DPD, and 90_PLUS. Paid-off loans,
default dispositions, RPL exits, and defect removals have explicit terminal states. Code 15
is a Release 47 loss-bearing default disposition. Code 16 is a censored RPL exit.

Roll rates use only observed records in adjacent calendar months. Exact duplicates collapse,
conflicting duplicates are removed, and gaps are skipped. I estimate pooled, vintage, FICO,
and original-LTV matrices with quarter expansion weights.

## Lifetime PD

I fit a weighted monthly logistic hazard model. The target is first entry into DEFAULTED in
the next adjacent month. The model uses loan age, current rate, estimated LTV, origination
FICO, origination LTV, origination DTI, origination rate, current credit state, vintage,
occupancy, loan purpose, and modification status.

To control training size, every event and every delinquent non-event is retained. A stable
one-in-six sample is taken only from CURRENT development non-events, and retained rows receive
the inverse selection probability in addition to the quarter expansion weight. Validation is
not case-control sampled.

Monthly conditional hazards are converted to survival, marginal PD, and cumulative PD. Loan
age advances one month at each projection step. Curves stop at the disclosed remaining legal
term. For SICR, current and origination-reference curves use the same snapshot age and
remaining term. The reference curve resets risk covariates to their first observed baseline,
which avoids comparing a remaining-life PD with a full age-zero lifetime PD.

## Cure and LGD

A delinquency cure requires three consecutive adjacent CURRENT months within a 12-month
outcome horizon. I fit a weighted logistic cure model to observed development episodes and
evaluate it on episodes beginning after the development cutoff.

Conditional severity uses Freddie Mac's disclosed Actual Loss divided by Zero Balance
Removal UPB. Raw negative values and values above one remain in the audit data. The
fractional-logit severity fit uses resolved default dispositions with realized LGD in the
unit interval. Every disclosed loss is also reconciled to the six Release 47 components.

For a nonterminal delinquent snapshot row, expected LGD equals one minus predicted cure
probability, multiplied by conditional severity. CURRENT rows have no active delinquency
episode, so their input is conditional severity. DEFAULTED rows are already in default, so
cure is not applicable and their input is conditional severity.

## EAD

Reporting-date EAD equals interest-bearing principal plus non-interest-bearing principal plus
delinquent accrued interest. For terminal defaults, Zero Balance Removal UPB is used as the
principal basis when disclosed. This is a term-loan exposure and no revolving credit
conversion factor is applied.

For future ECL months, scheduled principal follows fixed-rate contractual amortization. Any
nonprincipal difference between scalar reporting EAD and current UPB is held as an explicit
add-on. The scheduled one-month balance is checked against later observed balances for clean,
unmodified validation rows.

## Staging and ECL

Stage 3 takes precedence and is assigned to an explicit default or a 90 DPD backstop. Stage 2
is assigned by a 30 DPD backstop, a current-to-origination lifetime PD ratio of at least 2.0,
or an absolute lifetime PD increase of at least 5 percentage points. Remaining loans are
Stage 1. These quantitative SICR settings are research choices, not universal regulatory
thresholds.

Stage 1 uses marginal default probability over the next 12 months. Stage 2 uses the remaining
lifetime curve. Stage 3 applies immediate credit-impaired loss. Each monthly amount is PD
times LGD times amortizing EAD, discounted at the loan's effective annual interest rate.

## Scenarios and backtest

The base scenario has 60 percent weight and leaves hazard odds unchanged. The upside has
20 percent weight and multiplies monthly hazard odds by 0.75. The downside has 20 percent
weight and multiplies odds by 1.60. These are transparent sensitivity scenarios, not a claim
that an external macroeconomic forecast was estimated from this panel.

The backtest freezes probability-weighted ECL at December 2012. It then scans only later
disclosed Actual Loss events for those same loans and discounts each event to the snapshot
date. The primary accuracy view aligns the loss window with the provision: 12 months for
Stage 1, remaining modeled lifetime for Stage 2, and the full resolution window for Stage 3.
It compares weighted provision with weighted realized loss overall and by impairment stage.

I also report an ultimate-loss stress diagnostic that uses every disclosed loss through
March 2026 for all stages. This is intentionally not labeled a Stage 1 accuracy result because
it compares a 12-month provision with more than 13 years of outcomes. Prepayment and censoring
are zero-loss outcomes unless Freddie Mac later discloses Actual Loss within the applicable
observation window.
