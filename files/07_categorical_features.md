# 07 — Categorical Features Analysis

All 8 categorical (object dtype) features with their value distributions and fraud rates.

---

## F2230 — Data/Reporting Period

⚠️ **DATA LEAKAGE — SEE [09_data_leakage.md](./09_data_leakage.md)**

| Value | Count | Suspicious | Suspicious % |
|-------|-------|-----------|-------------|
| Oct25 | 9,001 | 0 | **0.00%** |
| Sep25 | 48 | 48 | **100.00%** |
| Nov25 | 23 | 23 | **100.00%** |
| Dec25 | 10 | 10 | **100.00%** |

**Interpretation**: Likely the data reporting period or batch identifier. All suspicious accounts 
are from non-October batches, creating perfect temporal separation.

---

## F3886 — Account Type

| Value | Count | % of Total | Suspicious | Suspicious Rate |
|-------|-------|-----------|-----------|----------------|
| Savings | 5,956 | 65.6% | 76 | 1.28% |
| Current | 2,051 | 22.6% | 4 | 0.20% |
| MSME Micro | 337 | 3.7% | 0 | 0.00% |
| MSME Small | 242 | 2.7% | 0 | 0.00% |
| Staff Loans | 108 | 1.2% | 0 | 0.00% |
| Agri Adv | 96 | 1.1% | 0 | 0.00% |
| Term Deposit | 87 | 1.0% | 0 | 0.00% |
| MSME Medium | 71 | 0.8% | 1 | 1.41% |
| Corp Adv | 23 | 0.3% | 0 | 0.00% |
| Gold Loan | 23 | 0.3% | 0 | 0.00% |
| Retail Others | 22 | 0.2% | 0 | 0.00% |
| MSME LA TDR | 20 | 0.2% | 0 | 0.00% |
| Retail LA TDR | 19 | 0.2% | 0 | 0.00% |
| HL | 10 | 0.1% | 0 | 0.00% |
| All Others | 10 | 0.1% | 0 | 0.00% |
| ML | 4 | 0.0% | 0 | 0.00% |
| PL | 3 | 0.0% | 0 | 0.00% |

**Key Finding**: 93.8% of suspicious accounts are **Savings** accounts. Mule accounts 
predominantly use savings accounts — easy to open, less scrutiny than current accounts.

---

## F3888 — Account Opening Date

| Metric | Value |
|--------|-------|
| Unique values | 4,292 |
| Date range | 1900-01-03 to 2025-11-28 |
| Computed account age (mean) | 8.6 years |
| Computed account age (median) | 7.8 years |

**Note**: Some dates go back to 1900, suggesting data quality issues or placeholder values. 
The majority of accounts were opened in the last 15 years.

**Confirmed**: F3887 (account age in days) = reference date (Oct 2025) minus F3888. 
Correlation = 0.9999.

---

## F3889 — Account Scheme/Product

| Value | Count | Suspicious | Suspicious Rate | Likely Meaning |
|-------|-------|-----------|----------------|----------------|
| G365D | 7,544 | 72 | 0.95% | General 365-Day (regular account) |
| L365D | 397 | 5 | 1.26% | Loan 365-Day |
| L7D | 386 | 0 | 0.00% | Loan 7-Day |
| L180D | 313 | 3 | 0.96% | Loan 180-Day |
| L90D | 207 | 0 | 0.00% | Loan 90-Day |
| L31D | 148 | 1 | 0.68% | Loan 31-Day |
| L14D | 87 | 0 | 0.00% | Loan 14-Day |

**Key Finding**: 88.9% of suspicious accounts are on the **G365D** (general/regular) scheme. 
Short-term loan schemes (L7D, L14D, L90D) have **zero** suspicious accounts.

---

## F3890 — Branch Area Classification

| Value | Full Name | Count | Suspicious | Suspicious Rate |
|-------|----------|-------|-----------|----------------|
| R | Rural | 2,015 | 29 | **1.44%** |
| SU | Semi-Urban | 2,390 | 21 | 0.88% |
| U | Urban | 1,777 | 13 | 0.73% |
| M | Metro | 2,900 | 18 | 0.62% |

**Key Finding**: **Rural** areas have the highest suspicious rate (1.44%), 
more than **2x the metro rate** (0.62%). This could reflect weaker KYC enforcement 
in rural branches or targeted exploitation of rural accounts as mules.

---

## F3891 — Occupation

| Value | Count | % of Total | Suspicious | Suspicious Rate |
|-------|-------|-----------|-----------|----------------|
| selfemployed | 3,951 | 43.5% | 26 | 0.66% |
| salaried | 1,909 | 21.0% | 14 | 0.73% |
| student | 1,185 | 13.0% | 23 | **1.94%** |
| agriculture | 1,112 | 12.2% | 14 | **1.26%** |
| housewife | 660 | 7.3% | 3 | 0.45% |
| others | 169 | 1.9% | 0 | 0.00% |
| retired | 96 | 1.1% | 1 | 1.04% |

**Key Finding**: **Students** have the highest suspicious rate (1.94%) — nearly **3x** 
the rate of self-employed. Students may be targeted as mules due to financial vulnerability.
**Agriculture** workers are second highest (1.26%).

---

## F3892 — Gender

| Value | Count | % of Total | Suspicious | Suspicious Rate |
|-------|-------|-----------|-----------|----------------|
| M (Male) | 5,007 | 55.1% | 63 | **1.26%** |
| F (Female) | 1,416 | 15.6% | 13 | 0.92% |
| O (Other) | 61 | 0.7% | 0 | 0.00% |
| NaN (Missing) | 2,598 | 28.6% | 5 | 0.19% |

**Key Finding**: Males are 77.8% of suspicious accounts. Male accounts have a 
1.37x higher suspicious rate than female accounts.
Note: 28.6% of records have missing gender — these have a very low suspicious rate.

---

## F3893 — Customer Segment

| Value | Count | Suspicious | Suspicious Rate |
|-------|-------|-----------|----------------|
| RETAIL | 6,437 | 76 | **1.18%** |
| CORPORATE | 2,645 | 5 | 0.19% |

**Key Finding**: **RETAIL** accounts are 93.8% of suspicious accounts with 
a **6.2x higher** suspicious rate than corporate accounts.
This aligns with mule account patterns — individual retail accounts are 
easier to recruit and harder to monitor.
