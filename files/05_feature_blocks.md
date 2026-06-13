# 05 — Feature Block Analysis (Thematic Groupings)

Features are organized in numbered blocks that follow clear thematic patterns.
This suggests the dataset was constructed by concatenating different categories of banking features.

---

## Block Map

| Block Range | # Cols | Dominant Type | % | Likely Theme |
|-------------|--------|--------------|---|-------------|
| **F1–F99** | 99 | proportion (0–1) | 100% | Behavioral scores / risk proportions |
| **F100–F199** | 100 | proportion (0–1) | 88% | Behavioral scores (continued) |
| **F200–F299** | 100 | proportion (0–1) | 95% | Channel usage / distribution ratios |
| **F300–F399** | 100 | small float | 92% | Transaction velocity ratios (centered ~1.0) |
| **F400–F499** | 100 | small float | 96% | Peer comparison ratios |
| **F500–F599** | 100 | small float | 96% | Trend multipliers / deviation scores |
| **F600–F699** | 100 | constant (42%), binary (24%), large float (13%) | mixed | Product/channel flags + cumulative amounts |
| **F700–F799** | 100 | constant (42%), binary (29%), large float (15%) | mixed | Product/channel flags + amounts (continued) |
| **F800–F899** | 100 | constant (43%), binary (29%), large float (11%) | mixed | Product/channel flags + amounts (continued) |
| **F900–F999** | 100 | binary (36%), large float (28%), constant (27%) | mixed | Transaction channel flags + amounts |
| **F1000–F1099** | 100 | binary (47%), large float (40%) | mixed | Channel indicators + monetary volumes |
| **F1100–F1199** | 100 | binary (46%), large float (42%) | mixed | Channel indicators + volumes (continued) |
| **F1200–F1299** | 100 | binary (45%), large float (43%) | mixed | Channel indicators + volumes (continued) |
| **F1300–F1399** | 100 | binary (47%), large float (41%) | mixed | Channel indicators + volumes (continued) |
| **F1400–F1499** | 100 | binary (47%), large float (41%) | mixed | Channel indicators + volumes (continued) |
| **F1500–F1599** | 100 | large float (43%), binary (36%), count (11%) | mixed | Amounts + counts + flags |
| **F1600–F1699** | 100 | count (45%), large float (38%) | mixed | Transaction counts by type/channel |
| **F1700–F1799** | 100 | large float (44%), count (41%) | mixed | Amounts + counts |
| **F1800–F1899** | 100 | count (44%), large float (42%) | mixed | Counts + amounts |
| **F1900–F1999** | 100 | large float (44%), proportion (42%) | mixed | Volume proportions + amounts |
| **F2000–F2099** | 100 | large float (47%), proportion (47%) | equal | Volume ratios + amounts |
| **F2100–F2199** | 100 | proportion (46%), large float (46%) | equal | Proportions + volumes |
| **F2200–F2299** | 100 | large float (43%), small float (36%) | mixed | Derived ratios + aggregations |
| **F2300–F2399** | 100 | small float (51%), large float (45%) | mixed | Statistical aggregations |
| **F2400–F2499** | 100 | large float (54%), small float (43%) | mixed | Scored/weighted metrics |
| **F2500–F2599** | 100 | **negative float (76%)** | dominant | **Month-over-month changes / deltas** |
| **F2600–F2699** | 100 | **negative float (96%)** | dominant | **Change metrics / velocity deltas** |
| **F2700–F2799** | 100 | **negative float (97%)** | dominant | **Change metrics / z-scores** |
| **F2800–F2899** | 100 | count (67%), negative float (20%) | mixed | Transaction counts + some changes |
| **F2900–F2999** | 100 | **count (88%)** | dominant | **Raw transaction counts** |
| **F3000–F3099** | 100 | **count (90%)** | dominant | **Raw transaction counts (continued)** |
| **F3100–F3199** | 100 | count (41%), negative float (38%) | mixed | Counts + deviation metrics |
| **F3200–F3299** | 100 | negative float (75%), binary (18%) | mixed | Growth rates + flags |
| **F3300–F3399** | 100 | negative float (72%), binary (21%) | mixed | Growth rates + flags |
| **F3400–F3499** | 100 | negative float (88%) | dominant | Growth/change rates |
| **F3500–F3599** | 100 | negative float (88%) | dominant | Growth/change rates |
| **F3600–F3699** | 100 | negative float (88%) | dominant | Growth/change rates |
| **F3700–F3799** | 100 | negative float (86%) | dominant | Growth/change rates |
| **F3800–F3899** | 100 | mixed (neg 45%, prop 18%, large 11%) | mixed | **Account profile / metadata features** |
| **F3900–F3924** | 25 | binary (18 of 25) | dominant | **Alert/status flags + TARGET** |

---

## Visual Summary

```
FEATURE RANGE THEME MAP:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

F1 ────────── F299   │ PROPORTIONS (behavioral scores)
F300 ───────── F599   │ RATIOS (velocity, peer comparison)
F600 ───────── F1499  │ FLAGS + MONETARY AMOUNTS (channel/product)
F1500 ──────── F2199  │ COUNTS + PROPORTIONS + AMOUNTS
F2200 ──────── F2499  │ DERIVED SCORES & AGGREGATIONS
F2500 ──────── F2799  │ DELTAS / CHANGES (month-over-month)
F2800 ──────── F3099  │ RAW COUNTS (transaction counts)
F3100 ──────── F3799  │ GROWTH RATES & DEVIATIONS
F3800 ──────── F3899  │ ACCOUNT PROFILE & METADATA
F3900 ──────── F3924  │ ALERT FLAGS + TARGET VARIABLE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Implications for Feature Engineering

1. **Proportions (F1–F299)**: Already normalized, ready to use. May represent time-of-day distributions, channel usage patterns.

2. **Ratios (F300–F599)**: Values centered around 1.0 suggest peer/baseline comparisons. Values > 1 = above average, < 1 = below average.

3. **Flags + Amounts (F600–F1499)**: Heavy in constant columns (~42%). After removing constants, the binary flags indicate product/channel activity and the large floats are cumulative amounts.

4. **Deltas (F2500–F2799)**: These 300 features capture behavioral **changes over time** — critical for fraud detection since mule accounts often show sudden shifts in activity patterns.

5. **Raw Counts (F2800–F3099)**: Transaction counts that can reveal unusual activity volumes.

6. **Growth Rates (F3100–F3799)**: Percentage changes that normalize absolute counts — useful for comparing across different account sizes.
