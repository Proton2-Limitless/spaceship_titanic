# Spaceship Titanic: what improved the score

**Result:** public LB 0.79494 (Random Forest, 3 engineered feature blocks). Raw baseline: 0.7884.

## Setup
Metric: accuracy; classes ~50/50 (50.4% transported), so a majority-class guess scores 0.5036.
Decisions were made on grouped 5-fold CV repeated with 3 seeds (15 scores). Groups stay whole
because train and test share 0 groups, so the real task is predicting unseen groups.
The leaderboard was used only for two checks, never to choose features. Sub 1 CV-vs-LB gap
was 0.006, about one standard error (SE ≈ 0.006 at n = 4,277), so CV was trusted.
Keep rule, fixed before the results: a block is kept for a model if it raises its CV
accuracy by ≥ 0.002 (differences below that are noise; fold std ≈ 0.0095).

## What improved it
| Change | Logistic CV | RF CV | Verdict |
|---|---|---|---|
| Dummy (majority) | 0.5036 | | floor |
| Raw columns | 0.7863 | 0.7941 (LB 0.7884) | baseline |
| **+ cabin (Deck, CabinNum, Side)** | +0.0053 | **+0.0159** | **kept; only clear gain** |
| + rules (logical CryoSleep/spend fill) | 0.0000 | 0.0000 | kept (harmless) |
| + groupfill (HomePlanet from group) | +0.0001 | -0.0002 | kept (0 of 9,280 groups conflict) |
| Final RF | | 0.8088 (LB 0.79494) | submitted |

## What did not help (and what I learned)
- **log1p on spend:** -0.0125 for Logistic (tested in isolation). My heavy-tail explanation
  was tested (raw TotalSpend was neutral) and rejected; the true mechanism is untested.
- **rules:** exactly 0. The default imputers already gave the same values (all spend medians
  are 0; mode CryoSleep is False). **rules2** (fill CryoSleep=True for zero-spenders, 85.1%
  of them were asleep) could touch only 87 rows and scored -0.0007 for RF.
- **group size, family size, IsChild:** neutral or slightly negative for RF.
- **CabinBin (one-hot bins of 100):** +0.0046 for Logistic, -0.0033 for RF. Cabin number is a
  wave, not a slope (rate 0.36 at bin 4, 0.67 at bin 9), which explains why trees gained
  3x more from cabin than Logistic. Not separated from deck effects.
- **Leakage:** I expected random CV to beat grouped CV and measured no gap (-0.0013 / +0.0010).
  Grouped CV was kept on principle.

## Limitations
- Features chosen after seeing CV, so CV is mildly optimistic: final CV-LB gap was 0.014.
- The LB gain over baseline (+0.0065) is about one SE; the CV gain is the stronger evidence.
- Only Logistic and Random Forest were compared. XGBoost/LightGBM/CatBoost, tuning, and
  ensembling were not run.

## Next steps
Gradient-boosted models on the settled features, tuning with grouped CV, a Logistic+RF blend,
and separating cabin number from deck.