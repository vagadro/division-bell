# Do real agents record what they built on? (eval B)

10 hand-written situations, run through the real Researcher, Planner and Writer (Claude Haiku 5.5) in two chat turns each: the user states three facts, then asks for a plan and a draft. Cost $0.13. Nothing was told to the agents about this test. A first pilot on one situation used an open-ended request that needed a number the user never gave, so the Researcher asked for it and wrote no plan; the requests were then reworded to be answerable from the stated facts (the pilot is not counted).

**Truth.** A decision or draft relied on a stated fact if its text contains that fact's value. This is a lower bound: an item can rely on a fact without repeating the value.
**Recorded.** The item is in blast_radius(fact): a DERIVED_FROM path leads back to the fact.

- Decisions and drafts written by the agents: **18**. Of these, **0** recorded no source at all.
- Situations where a decision was written: **9 of 10**; where a draft was written: **9 of 10**.
- Stated facts the Researcher saved: **29 of 30**.
- Dependencies that show in the text: **42**. Recorded: **42** (**100%**), missed: 0 (0%).
- Items flagged that do not repeat the fact's value: 10 of 52 (19%). These may be real dependencies stated in other words, or links that are too generous.

## Per situation

| Situation | Facts saved | Decisions and drafts | Dependencies in text | Recorded | Missed |
|---|---|---|---|---|---|
| bakery | 2/3 | 2 | 2 | 2 | 0 |
| trek | 3/3 | 2 | 6 | 6 | 0 |
| consulting | 3/3 | 2 | 3 | 3 | 0 |
| moving | 3/3 | 2 | 4 | 4 | 0 |
| podcast | 3/3 | 2 | 5 | 5 | 0 |
| sciencefair | 3/3 | 2 | 5 | 5 | 0 |
| release | 3/3 | 2 | 5 | 5 | 0 |
| garden | 3/3 | 2 | 6 | 6 | 0 |
| grant | 3/3 | 2 | 6 | 6 | 0 |
| tour | 3/3 | 0 | 0 | 0 | 0 |

## Limits

- Ten short situations written for this test, one model (Haiku 5.5), one run each. Real agents face longer and messier work.
- The truth rule is mechanical. It misses dependencies that are not repeated in the text, and it counts a repeated number as reliance even when it is a coincidence.
- Each situation has one user and a fresh graph, so this measures recording, not recall across many sessions.
