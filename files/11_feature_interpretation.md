# 11 — Feature Interpretation (Reverse-Engineering Anonymized Columns)

Attempt to determine what the anonymized columns (F1–F3924) represent, 
based on statistical properties, value patterns, and banking domain knowledge.

---

## Confirmed Identities (High Confidence)

| Feature | Identity | Confidence | Evidence |
|---------|----------|------------|---------|
| **F3888** | Account Opening Date | ✅ Very High | Date format ("8-1-2011"), range 1900–2025 |
| **F3887** | Account Age (days) | ✅ Very High | **0.9999 correlation** with (Oct 2025 - F3888) |
| **F3894** | Customer Age (years) | ✅ High | Range [-2, 94], median=34, corr=0.40 with account age |
| **F3886** | Account Type | ✅ Exact | Literal values: Savings, Current, MSME, etc. |
| **F3889** | Account Scheme/Product | ✅ Exact | G365D, L365D, L180D, L90D, L31D, L14D, L7D |
| **F3890** | Branch Area Classification | ✅ Exact | R=Rural, M=Metro, SU=Semi-Urban, U=Urban |
| **F3891** | Occupation | ✅ Exact | selfemployed, salaried, student, agriculture, etc. |
| **F3892** | Gender | ✅ Exact | M, F, O |
| **F3893** | Customer Segment | ✅ Exact | RETAIL, CORPORATE |
| **F2230** | Data/Reporting Period | ✅ High | Oct25, Sep25, Nov25, Dec25 |
| **F3912** | Fraud/Mule Alert Flag | ✅ Very High | Binary, 0.97 correlation with target, near-identical distribution |
| **F3924** | Target: Suspicious Account | ✅ Given | Binary 0/1, defined in problem statement |

---

## Probable Identities (Medium Confidence)

| Feature | Probable Identity | Evidence |
|---------|------------------|---------|
| F115 | Risk score / behavioral proportion | [0,1], 97 unique, higher in suspicious (0.72 vs 0.59) |
| F321 | Transaction velocity ratio | [0,10.6], centered ~1.0, correlates with F531 (0.50) |
| F527 | Peer comparison / baseline ratio | [0,48.4], centered ~1.0 |
| F531 | Transaction diversity/complexity score | [0,20.96], correlates with F321 |
| F670 | High-risk alert flag | Binary, 2.6x enrichment in suspicious |
| F1692 | Flagged event count | Integer [0,14], low values |
| F2082 | Rare/special transaction proportion | [0,1], 95th%=0.16, zero for all suspicious |
| F2122 | Digital/online channel ratio | [0,1], 1235 unique, very low for suspicious |
| F2582 | Month-over-month behavioral change | [-0.88,18.89], centered ~0 |
| F2678 | Transaction amount deviation / volatility | [-0.91, 1.6M], massive range, 30x lower for suspicious |
| F2737 | Balance change metric | [-0.94, 1707.6], centered near 0 |
| F2956 | Total transaction count | [0, 11548], integer-like, 0.98 corr with F3043 |
| F3043 | Cross-channel/period transaction count | [0, 21819], near-duplicate of F2956 |
| F3836 | Total balance/net throughput | [-20B, +16B], sign reversal between classes |
| F3908 | Product/verification flag | Binary, 30.6% have it, +0.097 corr |

---

## Feature Block Themes (Inferred)

### F1–F299: Behavioral Scores (Proportions 0–1)
**478 features** with values strictly in [0, 1].

These likely represent:
- Time-of-day transaction distributions (% of transactions at night, weekend, etc.)
- Channel usage proportions (% ATM, % mobile, % branch, etc.)
- Transaction type proportions (% cash, % transfer, % bill pay, etc.)
- Normalized risk scores from internal models
- Account activity percentiles

### F300–F599: Velocity / Comparison Ratios
**463 features** as small positive floats, many centered around 1.0.

These likely represent:
- Current period / previous period ratios (velocity)
- Account value / peer average ratios (deviation from norm)
- Transaction frequency compared to baseline
- Values > 1 = above average/increasing, < 1 = below average/decreasing

### F600–F1499: Product/Channel Flags + Monetary Amounts
~42% constant (no info), ~35% binary flags, ~20% large floats.

These likely represent:
- Binary indicators for specific products (has_loan, has_credit_card, etc.)
- Binary indicators for transaction channels used
- Cumulative transaction amounts by product/channel
- Many constants suggest products/channels not relevant to this account population

### F1500–F2199: Transaction Counts + Volume Proportions
Mix of counts, large floats, and proportions.

These likely represent:
- Raw transaction counts by type (credit, debit, cash, transfer)
- Transaction volumes (total amounts)
- Volume proportions (% of total by channel/type)

### F2200–F2499: Derived Statistical Aggregations
Mix of small and large floats.

These likely represent:
- Averages (average transaction size, average daily balance)
- Standard deviations (transaction amount variability)
- Percentile metrics
- Weighted composite scores

### F2500–F2799: Change / Delta Metrics (882 features with negatives)
**Largest block** — 300 features predominantly containing negative values.

These likely represent:
- Month-over-month changes in transaction counts
- Period-over-period changes in balances
- Z-scores (standard deviation from mean)
- Growth/decline rates
- Velocity changes (acceleration/deceleration)

**Why these matter for fraud**: Mule accounts often show sudden behavioral shifts — 
a dormant account suddenly receiving many transactions, or a stable account 
showing unusual transfer patterns. These delta features capture exactly those signals.

### F2800–F3099: Raw Transaction Counts
~400 count-like integer features.

These likely represent:
- Daily/weekly/monthly transaction counts
- Counts by channel (ATM, branch, online, mobile)
- Counts by type (credit, debit, transfer, cash withdrawal)
- Number of unique beneficiaries
- Login/access counts

### F3100–F3799: Growth Rates & Deviation Scores
~700 features, mostly negative-capable floats.

These likely represent:
- Percentage change metrics (week-over-week, month-over-month)
- Normalized deviation scores
- Z-scores and percentile ranks
- Risk scoring outputs from multiple internal models

### F3800–F3899: Account Profile & Metadata
Mixed types — this is where the identified features live.

Known: Account type, date, scheme, area, occupation, gender, segment, age.

### F3900–F3924: Alert/Status Flags + Target
18 binary features + target variable.

Known: F3912 = fraud flag (leakage), F3911 = constant, F3924 = target.
F3895–F3907 and F3915–F3923 are one-hot encoded categorical groups.

---

## One-Hot Encoded Groups

| Group | Features | Size | Probable Meaning |
|-------|----------|------|-----------------|
| F3895–F3907 | 13 binary columns | 13 | Month of some temporal attribute (month of opening, last transaction, etc.) |
| F3915–F3923 | 9 binary columns | 9 | Transaction channel primary type or account status category |

---

## Limitations

> **Important**: While we can infer general categories (proportions, counts, amounts, deltas), 
> the **exact business meaning** of individual features cannot be definitively determined 
> without the official data dictionary from Bank of India.
> 
> The anonymization is intentional for security. Feature numbers may not follow a 
> logical naming convention — they could be arbitrarily assigned or auto-generated 
> from a feature engineering pipeline.
>
> Our interpretations are based on:
> - Statistical properties (range, distribution, data type)
> - Cross-correlations between features
> - Banking fraud detection domain knowledge
> - Value patterns and block analysis
