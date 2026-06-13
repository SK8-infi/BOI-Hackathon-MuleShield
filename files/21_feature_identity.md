# Finding 21: Feature Identity — What Our 14 Model Features Actually Represent

**Date**: 2026-06-13  
**Purpose**: Reverse-engineer the banking meaning of our anonymized model features  

---

## 1. The Mule Account Signature (in Plain English)

Our model detects mule accounts using this pattern:

> **Mule accounts are newly opened, low-activity accounts with small balances and low transaction volumes,
> but they have a HIGH behavioral risk score (F162) — the only feature where mules score HIGHER than normal.**

This is exactly what anti-money-laundering literature describes: accounts opened quickly, used briefly
as pass-through channels, then abandoned.

---

## 2. Raw Features — Banking Interpretations

### F3898: Account Activity Tier / Product Engagement Level
| Property | Value |
|----------|-------|
| **Type** | int64, 8 unique values |
| **KS** | 0.517 (strongest raw feature) |
| **XGBoost Gain** | 0.42 (dominant feature) |
| **Missing** | 0% |

**Value Distribution:**
| Value | Count | Suspicious | Fraud Rate | Meaning |
|-------|-------|------------|------------|---------|
| 0 | 2,811 | 39 (48%) | **1.39%** | New / Inactive |
| 1 | 1,151 | 36 (44%) | **3.13%** | Low activity |
| 2 | 240 | 4 (5%) | 1.67% | Moderate activity |
| 3 | 4,755 | 2 (2%) | 0.04% | Normal active |
| 4+ | 125 | 0 (0%) | 0.00% | High engagement |

**Why it works**: 92% of mules are at tier 0 or 1 (new/low activity). Only 2 mules reached tier 3 (normal). The model essentially asks: *"Is this account barely used?"*

**Banking interpretation**: Product engagement score — how many products/services the customer uses (savings, credit card, insurance, etc.). Mule accounts only have a basic savings account.

---

### F162: Behavioral Risk Score / Anomaly Probability
| Property | Value |
|----------|-------|
| **Type** | float64, 0-1 scale, 101 unique values |
| **KS** | 0.354 |
| **Missing** | 36% |
| **Direction** | **HIGHER for suspicious** (the ONLY such feature) |

| Metric | Legitimate | Suspicious | Ratio |
|--------|-----------|------------|-------|
| Median | 0.62 | **0.88** | 1.42x |
| Mean | 0.64 | **0.80** | 1.25x |
| 75th %ile | 0.83 | **1.00** | 1.20x |

**Why it works**: This is likely a pre-computed risk/suspicion score from the bank's existing rule engine. Values are bounded 0-1 (probability). Mule accounts score significantly higher — median 0.88 vs 0.62 for legitimate.

**Banking interpretation**: Output from existing fraud detection rules or behavioral scoring model. The bank already has *some* signal, but it alone isn't sufficient (otherwise they wouldn't need ML).

---

### F1819: Cumulative Transaction Volume / Total Debit+Credit Sum
| Property | Value |
|----------|-------|
| **Type** | float64, 8,501 unique values |
| **KS** | 0.506 |
| **Missing** | 0.04% |

| Metric | Legitimate | Suspicious | Ratio S/L |
|--------|-----------|------------|-----------|
| Median | 1,483,515 | **228,016** | 0.015 |
| Mean | 110,654,596 | **658,049** | 0.006 |
| Max | 136,748,000,000 | 7,906,445 | 0.00006 |

**Why it works**: Mule account transaction volume is ~65x smaller. They handle small pass-through amounts, not the large sustained volume of real customers.

**Banking interpretation**: Total INR amount transacted (debits + credits) over the observation period. The enormous legitimate max (136B INR) suggests corporate accounts exist in the data.

---

### F3799: Peak/Maximum Account Balance
| Property | Value |
|----------|-------|
| **Type** | float64, 8,904 unique values |
| **KS** | 0.513 |
| **Missing** | 0% |

| Metric | Legitimate | Suspicious | Ratio S/L |
|--------|-----------|------------|-----------|
| Median | 1,736,020 | **237,968** | 0.014 |
| Mean | 95,128,198 | **727,907** | 0.008 |
| Min | 59 | **9,540** | — |

**Why it works**: Mule accounts never accumulate wealth. Maximum balance ~8x lower than legitimate.

**Banking interpretation**: Highest recorded balance in the account over its lifetime. Notable: suspicious min is 9,540 (not zero) — mule accounts do hold *some* money briefly.

---

### F1165: Average Individual Transaction Amount
| Property | Value |
|----------|-------|
| **Type** | float64, 3,803 unique values |
| **KS** | 0.500 |
| **Missing** | 0.02% |

| Metric | Legitimate | Suspicious | Ratio S/L |
|--------|-----------|------------|-----------|
| Median | 210,000 | **40,000** | 0.019 |
| Mean | 8,881,233 | **123,778** | 0.014 |
| Min | 0 | **5,000** | — |

**Why it works**: Mules transact in small amounts to stay under regulatory thresholds. Median individual transaction is just Rs 40,000 vs Rs 2,10,000 for normal customers.

**Banking interpretation**: Average transaction size (possibly average debit amount). The Rs 5,000 minimum suggests small structuring. Fewer unique values (3,803) vs other features (~8,500) suggests this is computed/rounded.

---

### F1813: Lifetime Total Transaction Amount
| Property | Value |
|----------|-------|
| **Type** | float64, 8,841 unique values |
| **KS** | 0.515 |
| **Missing** | 0.04% |

| Metric | Legitimate | Suspicious | Ratio S/L |
|--------|-----------|------------|-----------|
| Median | 1,652,508 | **237,968** | 0.014 |
| Mean | 232,598,774 | **691,823** | 0.003 |

**Why it works**: Essentially the same signal as F1819 but computed differently (possibly over a different time window — lifetime vs recent period). Mule accounts have ~330x less lifetime volume.

**Banking interpretation**: Total transaction value over the entire account life. Similar to F1819 but with slight calculation differences.

---

### F3806: Minimum Balance / Account Floor
| Property | Value |
|----------|-------|
| **Type** | float64, 8,358 unique values |
| **KS** | 0.493 |
| **Missing** | 0% |

| Metric | Legitimate | Suspicious | Ratio S/L |
|--------|-----------|------------|-----------|
| Median | 619,253 | **79,433** | 0.013 |
| Mean | 27,367,543 | **254,444** | 0.009 |
| Min | 0 | **0** | — |

**Why it works**: Mule accounts have very low floor balances — they drain funds quickly rather than maintaining a cushion.

**Banking interpretation**: Minimum recorded balance (or average low balance). Some mules reach zero (drained completely).

---

### F3800: Current/Recent Account Balance
| Property | Value |
|----------|-------|
| **Type** | float64, 8,614 unique values |
| **KS** | 0.490 |
| **Missing** | 0% |

| Metric | Legitimate | Suspicious | Ratio S/L |
|--------|-----------|------------|-----------|
| Median | 891,448 | **128,784** | 0.014 |
| Mean | 48,699,583 | **369,835** | 0.008 |

**Why it works**: Similar to F3806 but slightly higher values — likely a different balance snapshot (end-of-period vs minimum).

**Banking interpretation**: Average daily balance or period-end balance.

---

## 3. Dependency Features (Used in Engineering)

### F3805: Total Account Value / Assets Under Management
| Metric | Legitimate | Suspicious | Ratio |
|--------|-----------|------------|-------|
| Median | 1,216,688 | **192,227** | 0.016 |
| Mean | 52,873,119 | **515,004** | 0.010 |
| KS | **0.516** | | |

**Banking interpretation**: Highest of all balance-related features. Likely total AUM or cumulative credit value. All suspicious accounts have min=7,613 (no zeros).

### F3811: Total Outflow / Debit Volume
| Metric | Legitimate | Suspicious | Ratio |
|--------|-----------|------------|-------|
| Median | 853,831 | **135,590** | 0.016 |
| Mean | 29,344,206 | **356,129** | 0.012 |
| KS | **0.513** | | |

**Banking interpretation**: Cumulative debit/withdrawal amount. Mule outflows are 80x smaller.

---

## 4. Engineered Features — What They Capture

| Feature | Formula | Banking Meaning | Why It Helps |
|---------|---------|----------------|-------------|
| **F162_div_F3898** | Risk Score / Activity Tier | Risk-per-engagement | Amplifies: high risk + low activity = strong mule signal |
| **max_value_top8** | max(8 stable features) | Peak across all metrics | If the BEST number is still low, it's suspicious |
| **F3898_div_F3805** | Activity Tier / Total Value | Tier normalized by wealth | Low tier despite having some value = suspicious |
| **missing_count_F0_F500** | NULL count in F0-F499 | Data completeness | Mules have 97.5% data vs 63% for legit (more complete!) |
| **F3811_div_F3805** | Total Outflow / Total Value | Debit-to-credit ratio | How much flows OUT vs came IN — layering indicator |
| **F3898_div_F3811** | Activity Tier / Total Outflow | Engagement per outflow | Low engagement despite outflows = pass-through behavior |

---

## 5. Summary Table — All 14 Features

| # | Feature | Banking Meaning | Range | Direction | KS | Role |
|---|---------|----------------|-------|-----------|-----|------|
| 1 | **F3898** | Account activity tier | 0-57 (int) | Low = Mule | 0.517 | **Dominant** |
| 2 | **F162** | Behavioral risk score | 0-1 | **High = Mule** | 0.354 | Risk signal |
| 3 | **F1819** | Transaction volume | 0-137B | Low = Mule | 0.506 | Volume signal |
| 4 | **F3799** | Peak balance | 59-63B | Low = Mule | 0.513 | Balance signal |
| 5 | **F1165** | Avg transaction size | 0-7.7B | Low = Mule | 0.500 | Size signal |
| 6 | **F1813** | Lifetime txn amount | 0-338B | Low = Mule | 0.515 | Volume signal |
| 7 | **F3806** | Minimum balance | 0-39B | Low = Mule | 0.493 | Balance floor |
| 8 | **F3800** | Current balance | 0-39B | Low = Mule | 0.490 | Balance snapshot |
| 9 | F162_div_F3898 | Risk per engagement | ratio | High = Mule | — | Interaction |
| 10 | max_value_top8 | Peak across metrics | monetary | Low = Mule | — | Aggregation |
| 11 | F3898_div_F3805 | Tier per total value | ratio | Low = Mule | — | Normalization |
| 12 | missing_count_F0_F500 | Data completeness | 0-500 | **Low = Mule** | — | Metadata signal |
| 13 | F3811_div_F3805 | Debit/credit ratio | ratio | — | — | Flow pattern |
| 14 | F3898_div_F3811 | Tier per outflow | ratio | Low = Mule | — | Engagement/flow |

---

## 6. Overlooked Feature Scan — Conclusion: No Real Hidden Gems

### What We Found
Scanned **3,885 features** not already in our model. The top-KS features (F518, F515, F413, etc.) all have KS > 0.9 — but they are **statistical artifacts**:

| Feature | KS | Miss% Legit | Miss% Susp | NonNull Susp | Verdict |
|---------|-----|-------------|------------|--------------|---------|
| F518 | 0.994 | 90.3% | **98.8%** | **1** | Artifact |
| F515 | 0.994 | 90.3% | **98.8%** | **1** | Artifact |
| F413 | 0.991 | 90.3% | **98.8%** | **1** | Artifact |
| F416 | 0.991 | 90.3% | **98.8%** | **1** | Artifact |
| F313 | 0.936 | 94.6% | **98.8%** | **1** | Artifact |
| F316 | 0.932 | 94.6% | **98.8%** | **1** | Artifact |

**Every single one has only 1 non-null suspicious account!** The high KS is because comparing 1 sample vs 500+ samples always looks "different." XGBoost exploits the missingness pattern (98.8% missing for mules vs 90% for legit), but `missing_count_F0_F500` already captures this signal more robustly.

### Bottom Line
> **Our 14 features already capture all discoverable signal.** There are no hidden gems in the remaining 3,885 features — only statistical noise and missingness artifacts.
