# Blast-radius accuracy

**Data.** 16 scenarios from 16 domains, 246 items (facts, decisions, drafts, procedures; 78 are decoys on the same topic that nothing relied on). The wording was written by a language model for hand-designed scenarios. 98 questions ("this fact is wrong, what relied on it?") with 329 items that truly relied on it: 122 at 1 hop, 121 at 2, 86 at 3 or more. 114 of them (35%) share at most one content word with the wrong fact.

**Reading the table.** Precision = of the items flagged, the share that really relied on the wrong fact. Recall = of the items that relied on it, the share flagged. F1 combines them. In brackets: 95% interval from resampling the 16 scenarios. "Innocent per real" = false alarms for every item correctly flagged.

| Method | Precision | Recall | F1 | Innocent per real |
|---|---|---|---|---|
| Graph (blast_radius) | 1.00 | 1.00 | 1.00 | 0.0 |
| Graph, 1 hop only | 1.00 | 0.37 (0.33-0.41) | 0.54 (0.49-0.59) | 0.0 |
| Flag every later item in the scenario | 0.31 (0.28-0.35) | 1.00 | 0.48 (0.44-0.51) | 2.2 |
| Vector similarity, whole memory | 0.42 (0.36-0.49) | 0.47 (0.37-0.56) | 0.44 (0.38-0.50) | 1.4 |
| Vector similarity, later items of the scenario | 0.53 (0.46-0.60) | 0.72 (0.63-0.81) | 0.61 (0.56-0.66) | 0.9 |
| Keyword overlap, later items of the scenario | 0.48 (0.42-0.54) | 0.85 (0.78-0.91) | 0.61 (0.57-0.66) | 1.1 |
| LLM reads all the notes (Haiku, mean of 3 runs) | 1.00 (0.99-1.00) | 0.73 (0.64-0.81) | 0.84 (0.78-0.89) | 0.0 |

The graph has no interval because its answers equal the key on every scenario. The 3 reader runs alone scored F1 0.84-0.85 (precision 0.99-1.00, recall 0.72-0.74). Similarity cutoffs are chosen leave-one-scenario-out (each scenario is scored with the cutoff that is best on the other 15). Even the best single cutoff, picked with the answers in view, only reaches F1 0.46 (vector, whole memory), 0.62 (vector, later items) and 0.61 (keyword, later items).

## Per scenario (F1)

| Scenario | Questions | Graph | 1 hop | Flag all | Vector (all) | Vector (later) | Keyword | LLM reader |
|---|---|---|---|---|---|---|---|---|
| clinic (healthcare operations) | 5 | 1.00 | 0.54 | 0.51 | 0.36 | 0.70 | 0.52 | 0.50 |
| pgmigrate (software engineering) | 6 | 1.00 | 0.54 | 0.44 | 0.33 | 0.49 | 0.70 | 0.73 |
| river (nonprofit fundraising) | 8 | 1.00 | 0.57 | 0.62 | 0.37 | 0.63 | 0.69 | 0.83 |
| wedding (event planning) | 5 | 1.00 | 0.61 | 0.42 | 0.40 | 0.56 | 0.55 | 0.70 |
| retail (retail inventory) | 5 | 1.00 | 0.59 | 0.49 | 0.45 | 0.71 | 0.59 | 0.85 |
| course (university teaching) | 5 | 1.00 | 0.63 | 0.33 | 0.35 | 0.32 | 0.57 | 0.93 |
| bikelane (local government) | 7 | 1.00 | 0.60 | 0.43 | 0.23 | 0.42 | 0.58 | 0.65 |
| litreview (medical research) | 8 | 1.00 | 0.62 | 0.44 | 0.47 | 0.69 | 0.67 | 0.97 |
| foodtruck (small business) | 6 | 1.00 | 0.67 | 0.35 | 0.39 | 0.51 | 0.50 | 0.96 |
| cloudcost (cloud cost management) | 7 | 1.00 | 0.49 | 0.52 | 0.50 | 0.65 | 0.75 | 0.94 |
| newsroom (journalism) | 5 | 1.00 | 0.67 | 0.40 | 0.69 | 0.64 | 0.56 | 0.92 |
| fitnessapp (product management) | 6 | 1.00 | 0.57 | 0.40 | 0.67 | 0.73 | 0.51 | 0.96 |
| warehouse (logistics) | 6 | 1.00 | 0.35 | 0.58 | 0.32 | 0.63 | 0.69 | 0.87 |
| renovation (home renovation) | 5 | 1.00 | 0.43 | 0.52 | 0.33 | 0.54 | 0.51 | 0.71 |
| claims (insurance operations) | 7 | 1.00 | 0.39 | 0.52 | 0.58 | 0.65 | 0.54 | 0.93 |
| mlpaper (machine learning research) | 7 | 1.00 | 0.57 | 0.53 | 0.56 | 0.72 | 0.77 | 0.78 |

## Recall by how far down the chain, and by shared wording

| Method | 1 hop | 2 hops | 3+ hops | Shares at most 1 word with the wrong fact | Shares more |
|---|---|---|---|---|---|
| Graph (blast_radius) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Graph, 1 hop only | 1.00 | 0.00 | 0.00 | 0.25 | 0.44 |
| Flag every later item in the scenario | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Vector similarity, whole memory | 0.65 | 0.36 | 0.35 | 0.17 | 0.62 |
| Vector similarity, later items of the scenario | 0.84 | 0.63 | 0.67 | 0.47 | 0.85 |
| Keyword overlap, later items of the scenario | 0.93 | 0.82 | 0.79 | 0.57 | 1.00 |
| LLM reads all the notes (Haiku, mean of 3 runs) | 0.85 | 0.69 | 0.62 | 0.52 | 0.84 |

## What the other methods get wrong

**Vector similarity, later items of the scenario**

*flagged although it did not rely on the wrong fact:*
- wrong: "Claim 8841: the customer reports a burst pipe on 3 June and notified the insurer on 9 July." -> "The plumber's report for claim 8841 states that the pipe failed suddenly."
- wrong: "Holiday forecast for knit goods: 9,000 units." -> "Knit goods on hand: 3,100 units."
- wrong: "The 2022 Marlowe trial enrolled 214 patients and reported a 31% reduction in migraine days at 12 weeks." -> "A 2019 trial of a different compound reported a 22% reduction in migraine days."
*missed although it relied on the wrong fact:*
- wrong: "The barn's licence caps total attendance at 110, and that count includes staff." -> "Chasing late RSVPs: Send a text to anyone who hasn't replied on day one.; If still no answer, give them a call on day three.; On day five, ask the par"
- wrong: "Tacos are about 70% of what we sell at the weekend market, and the sides barely move." -> "Cancelling the standing order of paper boats for side dishes, so we stop paying for boats nobody's using."
- wrong: "In the exit survey, 58% of students reported that the weekly problem sets took more than six hours." -> "Moderating marks across TAs: Have each TA mark the same sample of 10 scripts independently.; Compare the marks each TA gave to every script side by si"

**LLM reads all the notes (run 1)**

*flagged although it did not rely on the wrong fact:*
- wrong: "The bus depot on Harbor Road will close, and 140 jobs go with it. The transit authority confirmed the closure this week, ending a run that began in 19" -> "The closure puts nearly half of our members at risk," said the union spokesperson. "We are not going to let these jobs disappear quietly."
*missed although it relied on the wrong fact:*
- wrong: "Walk-in evening peak: avg wait about 75 min (front-desk logs, last quarter)." -> "Decided: open Saturdays 8 to 12 on a three-month trial, 13 Saturdays."
- wrong: "Prod is on Postgres 12; EOL in November, security patches stop then." -> "Managed large tier it is. xlarge not needed. Large covers peak writes with room to spare, so no reason to pay for the bigger box."
- wrong: "A survey of the 14 km Alder Brook stretch identified nine culverts that block fish passage." -> "Please find enclosed our application for support towards the Alder Brook barrier removal project. We respectfully submit it for the Fund's considerati"

## If agents record the links badly (simulation)

The graph can only follow links that were recorded. These rates are assumptions, not measurements of real agents. Each row averages 5 random draws over the same 98 questions. *Forget* drops that share of the recorded links. *Invent* adds, with that probability per item, one link to an earlier item the item did not rely on (preferring the same topic, so decoys get cited).

| Links | Rate | Precision | Recall | F1 |
|---|---|---|---|---|
| forget | 10% | 1.00 | 0.80 | 0.89 |
| forget | 25% | 1.00 | 0.64 | 0.78 |
| invent | 10% | 0.88 | 1.00 | 0.94 |
| invent | 25% | 0.79 | 1.00 | 0.88 |

## Limits

- The wording is model-written for 16 hand-designed situations, not real agent traffic, and the LLM reader is the same model family that wrote it. The dependency key is known by construction, and a different author could disagree on a few borderline items.
- The graph answers from recorded links, so its score on perfect links is a correctness check of the traversal. A plain list of parent ids walked in code would give the same answer; the evidence here is that similarity and reading cannot recover dependency, and that the traversal is exact.
- The key counts every item downstream of a wrong fact as relying on it, however many steps away, which is what the traversal returns. A person might call a distant item (a procedure three steps down) unaffected, so read the graph's flags as 'review these', a safe superset, not 'these are certainly wrong'.
- The key was audited once after the first run, by reading where the reader consistently disagreed with it. One flaw was found and fixed, which removed one of the reader's apparent false alarms, so the fix went against the graph. Two borderline cases remain, both between sibling drafts of the same story.
- 16 scenarios is a small sample (see the intervals). The LLM reader is one model (Haiku 5.5) at medium effort; a bigger model would probably score higher. Its cost grows with the size of the memory it must read (each question sent the whole notebook here, about 15 notes); the graph query does not. That scaling was not measured here.
