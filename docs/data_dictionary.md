# Freddie Mac Release 47 data contract

The supplied archives use the Standard Single-Family Loan-Level Dataset Release 47 layout,
effective July 2026. This contract is checked against the observed width of every file before
any position is interpreted.

Authoritative references:

- [Release 47 General User Guide](https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf)
- [Release 47 disclosure changes](https://www.freddiemac.com/fmac-resources/research/pdf/disclosure-changes-summary.pdf)
- [Freddie Mac release notes](https://www.freddiemac.com/fmac-resources/research/pdf/release_notes.pdf)

## Origination file, 31 fields

| Position | Attribute |
|---:|---|
| 1 | Classic FICO |
| 2 | First Payment Date |
| 3 | First Time Homebuyer Indicator |
| 4 | Maturity Date |
| 5 | Metropolitan Statistical Area or Metropolitan Division |
| 6 | Mortgage Insurance Percentage |
| 7 | Number of Units |
| 8 | Occupancy Status |
| 9 | Original Combined Loan-to-Value |
| 10 | Original Debt-to-Income Ratio |
| 11 | Original UPB |
| 12 | Original Loan-to-Value |
| 13 | Original Interest Rate |
| 14 | Channel |
| 15 | Prepayment Penalty Indicator |
| 16 | Amortization Type |
| 17 | Property State |
| 18 | Property Type |
| 19 | Postal Code |
| 20 | Loan Identifier |
| 21 | Loan Purpose |
| 22 | Original Loan Term |
| 23 | Number of Borrowers |
| 24 | Seller Name |
| 25 | Super Conforming Flag |
| 26 | Pre-HARP Loan Sequence Number |
| 27 | Special Eligibility Program |
| 28 | HARP Indicator |
| 29 | Property Valuation Method |
| 30 | Interest Only Indicator |
| 31 | VantageScore 4.0 |

## Monthly performance file, 35 fields

| Position | Attribute |
|---:|---|
| 1 | Loan Identifier |
| 2 | Period |
| 3 | Current Actual UPB |
| 4 | Current Loan Delinquency Status |
| 5 | Loan Age |
| 6 | Remaining Months to Legal Maturity |
| 7 | Underwriting Defect and Major Servicing Defect Settlement Date |
| 8 | Modification Flag |
| 9 | Zero Balance Code |
| 10 | Zero Balance Effective Date |
| 11 | Current Interest Rate |
| 12 | Current Non-Interest Bearing UPB |
| 13 | Due Date of Last Paid Installment |
| 14 | MI Recoveries |
| 15 | Net Sales Proceeds |
| 16 | Non-MI Recoveries |
| 17 | Total Expenses |
| 18 | Legal Costs |
| 19 | Maintenance and Preservation Costs |
| 20 | Taxes and Insurance |
| 21 | Miscellaneous Expenses |
| 22 | Actual Loss |
| 23 | Cumulative Modification Costs |
| 24 | Interest Rate Step Indicator |
| 25 | Payment Deferral Flag |
| 26 | Estimated Loan-to-Value |
| 27 | Zero Balance Removal UPB |
| 28 | Delinquent Accrued Interest |
| 29 | Delinquency Due to Disaster |
| 30 | Borrower Assistance Plan |
| 31 | Current Period Modification Costs |
| 32 | Current Interest Bearing UPB |
| 33 | Mortgage Insurance Cancellation Indicator |
| 34 | Servicer Name |
| 35 | Bankruptcy Cramdown Costs |

## Release 47 interpretation decisions

Release 47 discloses recoveries and gains as negative amounts and expenses and losses as
positive amounts. Actual Loss is validated as:

```text
Actual Loss = Zero Balance Removal UPB
            + Net Sales Proceeds
            + Delinquent Accrued Interest
            + Total Expenses
            + MI Recoveries
            + Non-MI Recoveries
```

Total Expenses is column 17 and already represents the expense total. Columns 18 through 21
are its disclosed components. A reconciliation must use column 17 or the component sum, never
both.

Actual Loss is populated for Zero Balance Codes 02, 03, 09, and 15, subject to the guide's
availability rules. Code 15 remains a loss-bearing Note Sale or non-performing loan sale and
is treated as default. Code 16 is a reperforming loan sale or securitization with no disclosed
loss and is treated as right-censored.

Delinquency status is a string. Values 00, 01, and 02 mean current, 30 to 59 days past due,
and 60 to 89 days past due. Numeric values 03 and above are grouped into 90 or more days past
due for the Phase 0 matrix. RA means REO acquisition and XX means unavailable.

Through September 2025, status 00 can also describe an accounting cycle before the first
scheduled payment. Loan age and First Payment Date are retained so later phases can identify
these records explicitly.

## Reporting-cycle caveat

The older accounting and default reporting cycles are offset for performance periods through
April 2019, inclusive. The 2005 to 2007 origination vintages have histories on both sides of
that date. The caveat therefore applies by performance period, not to each vintage as a whole.

