# 10 — Suspicious / Mule Account Profile

Deep dive into the 81 accounts labeled as suspicious (F3924 = 1).

---

## Overview

| Metric | Value |
|--------|-------|
| Total suspicious accounts | 81 |
| Percentage of dataset | 0.89% |
| Distribution across time periods | Sep25: 48, Nov25: 23, Dec25: 10 |

---

## Demographic Profile

### Account Type (F3886)
| Type | Count | % of Suspicious |
|------|-------|-----------------|
| **Savings** | **76** | **93.8%** |
| Current | 4 | 4.9% |
| MSME Medium | 1 | 1.2% |

→ Mule accounts are overwhelmingly **Savings accounts** (easy to open, less scrutiny).

---

### Account Scheme (F3889)
| Scheme | Count | % of Suspicious |
|--------|-------|-----------------|
| **G365D** | **72** | **88.9%** |
| L365D | 5 | 6.2% |
| L180D | 3 | 3.7% |
| L31D | 1 | 1.2% |

→ Regular/general accounts (G365D) dominate. Short-term loan schemes have zero mule accounts.

---

### Area (F3890)
| Area | Count | % of Suspicious | Rate in Area |
|------|-------|-----------------|-------------|
| **R (Rural)** | **29** | **35.8%** | **1.44%** |
| SU (Semi-Urban) | 21 | 25.9% | 0.88% |
| M (Metro) | 18 | 22.2% | 0.62% |
| U (Urban) | 13 | 16.0% | 0.73% |

→ **Rural** branches contribute the most suspicious accounts both in absolute count 
and rate. Metro areas have the lowest risk.

---

### Occupation (F3891)
| Occupation | Count | % of Suspicious | Rate in Group |
|-----------|-------|-----------------|--------------|
| selfemployed | 26 | 32.1% | 0.66% |
| **student** | **23** | **28.4%** | **1.94%** |
| salaried | 14 | 17.3% | 0.73% |
| agriculture | 14 | 17.3% | **1.26%** |
| housewife | 3 | 3.7% | 0.45% |
| retired | 1 | 1.2% | 1.04% |
| others | 0 | 0.0% | 0.00% |

→ **Students** have the highest risk rate (1.94%), nearly **3x self-employed**.
Students + agriculture account for 45.7% of mule accounts.

---

### Gender (F3892)
| Gender | Count | % of Suspicious |
|--------|-------|-----------------|
| **M (Male)** | **63** | **77.8%** |
| F (Female) | 13 | 16.0% |
| Missing | 5 | 6.2% |
| O (Other) | 0 | 0.0% |

→ Males are heavily overrepresented among suspicious accounts.

---

### Customer Segment (F3893)
| Segment | Count | % of Suspicious | Rate in Segment |
|---------|-------|-----------------|----------------|
| **RETAIL** | **76** | **93.8%** | **1.18%** |
| CORPORATE | 5 | 6.2% | 0.19% |

→ RETAIL accounts are 6.2x more likely to be suspicious than corporate accounts.

---

## Typical Mule Account Profile

Based on the data, the **highest-risk profile** is:

```
┌─────────────────────────────────────────┐
│          MULE ACCOUNT ARCHETYPE         │
├─────────────────────────────────────────┤
│  Account Type:    Savings               │
│  Scheme:          G365D (General)       │
│  Segment:         RETAIL                │
│  Branch Area:     Rural                 │
│  Occupation:      Student               │
│  Gender:          Male                  │
│  Account Age:     ~108 days (median)    │
│  Customer Age:    ~33 years (median)    │
│  F670 flag:       23.5% have it set     │
│  F2678 value:     Very low (~11 avg)    │
│  F2956 count:     Low (~58 avg)         │
└─────────────────────────────────────────┘
```

---

## Behavioral Indicators (from Key Features)

Comparing suspicious vs legitimate accounts on key numeric features:

| Feature | Legit (mean) | Suspicious (mean) | Ratio | Pattern |
|---------|-------------|-------------------|-------|---------|
| F115 (risk score) | 0.59 | **0.72** | 1.22x | Higher risk scores |
| F670 (flag) | 9.0% | **23.5%** | 2.6x | More alerts |
| F2082 (rare txn) | 0.021 | **0.000** | 0x | Zero rare transaction activity |
| F2122 (digital) | 0.046 | **0.005** | 0.11x | Almost no digital channel usage |
| F2678 (amount) | 351.25 | **11.41** | 0.03x | Tiny transaction deviation |
| F2956 (count) | 133.46 | **58.38** | 0.44x | Much fewer transactions |
| F3836 (balance) | -48.9M | **+1.49M** | sign flip | Net inflow instead of outflow |

### Behavioral Summary
Suspicious/mule accounts tend to:
1. ✅ Have **higher risk scores** (F115)
2. ✅ Trigger more **alert flags** (F670)
3. ❌ Have **zero** rare/special transaction types (F2082)
4. ❌ Use **almost no** digital channels (F2122)
5. ❌ Have **much fewer** overall transactions (F2956)
6. ❌ Show **minimal** transaction amounts (F2678)
7. 🔄 Show **net inflows** vs net outflows (F3836) — consistent with receiving fraudulent funds
