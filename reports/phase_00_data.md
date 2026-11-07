# Phase 0 report: data foundation

HingMix v0.1 is built: 458,621 train, 24,138 val, 766 test_a and 1,160 test_b rows.

| Acceptance criterion | Result |
| --- | --- |
| `make dataset` builds from verified downloads | Pass |
| HingBERT-LID at least 95% on gold | Pass, 99.58% |
| 300-token hand check logged | Done, 89.3% (`results/p0_lid_handcheck.csv`) |
| Zero train/test near-duplicates at Jaccard 0.8 | Pass, 0 for train and val against both tests |
| `MANIFEST.json` and datasheet draft | Pass |
| `00_explore_sources.ipynb` | Pass |

## Findings
- Scope was reduced from 3M to about 0.5M sentences for time. The PRD was updated.
- The hand check exposed foreign-language tweets, so a filter was added. It removed 10,488 rows.
- Test sets are chat-heavy and shorter: median 12 and 15.5 tokens, against 18 in train.
- Tagger noise on short Hindi abbreviations may affect switch-point results in later phases.
- The `Conversations` folder is the English CMU data and is unused. The Hinglish version replaces it.
