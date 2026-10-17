# Lifecycle checks

29 different rules, each checked on up to 30 templated companies (the shapes differ, the wording does not matter here): supersede, retract, confirm, recall and correct, and a sharing story between two users. Checks with nothing to check in a company are skipped, not counted. **954 of 954 evaluations passed.** Tenant isolation is covered by tests/test_tenancy.py.

| Check | Passed |
|---|---|
| supersede flags exactly the items built on the fact | 30/30 |
| supersede: flagged() says 'changed' and names the new fact | 27/27 |
| supersede does not count as a mistake | 30/30 |
| recall shows the new value, not the old one | 30/30 |
| retract flags exactly the items built on the fact | 30/30 |
| retract counts one mistake for the author | 30/30 |
| retract: flagged() says 'wrong' for every item built on it | 28/28 |
| an item built on a changed AND a wrong fact keeps both reasons | 11/11 |
| cannot build a fact on a retracted fact | 30/30 |
| cannot build a decision on a flagged item | 28/28 |
| cannot retract a fact twice | 30/30 |
| cannot supersede a retracted fact | 30/30 |
| another user cannot retract the fact | 30/30 |
| confirm works on an item flagged only 'changed' | 46/46 |
| confirm is refused on an item flagged 'wrong' | 131/131 |
| after confirm: changed-only items active, wrong items still flagged | 30/30 |
| recall leaves out retracted, superseded and flagged facts | 30/30 |
| recall shows confirmed facts again | 23/23 |
| correct flags exactly the items built on the fact | 30/30 |
| correct: new fact active, about the same entity, CORRECTS the old one (now retracted) | 30/30 |
| correct counts one mistake for the author | 30/30 |
| other user cannot see a private fact | 30/30 |
| other user cannot build on a private fact | 30/30 |
| other user cannot retract or supersede a shared fact | 30/30 |
| retract: other user's private items are counted, not shown | 30/30 |
| retract: the handoff recipient is listed to notify | 30/30 |
| retract: the report does not name the author of a hidden item | 30/30 |
| retract: the hidden item's author finds it in its own flagged() | 30/30 |
| other user's flagged() shows both items as 'wrong' | 30/30 |
