# HingMix v0.1 datasheet (draft)

Real-only, Roman-script Hinglish for next-word prediction. No synthetic rows yet.

| Split | Rows | Tokens | Source | Genre |
| --- | --- | --- | --- | --- |
| train | 458,621 | 10,203,635 | L3Cube-HingCorpus | tweet |
| val | 24,138 | 536,268 | L3Cube-HingCorpus | tweet |
| test_a | 766 | 11,358 | CMU Hinglish DoG test | chat |
| test_b | 1,160 | 19,379 | PHINC, seeded 10% | translation_pair |

Files are Parquet with one row per message. Each row has aligned `tokens` and `lid` lists.

## Build
`make dataset` verifies checksums, reads the first 550,000 lines of the shuffled HingCorpus file,
normalises, keeps 3 to 60 tokens, tags with HingBERT-LID, keeps code-mixed messages, removes
foreign-language text from train and val, removes train rows near-duplicating a test row, and splits.
Removed as foreign: 10,488. Near-duplicates removed: 0.

## Tagger quality
- HingLID test gold: 99.58% token accuracy on 193,547 tokens. Gold labels may not be independent.
- Hand check on 300 raw HingCorpus tokens: 89.3%, or about 93.7% excluding foreign-language tokens.
- Short Hindi abbreviations (`h`, `b`, `kr`) are often tagged `en`, creating false switches.
- The model has no `other` class. `other` is assigned by rule to non-words and special tokens.

## Known limitations
- Some foreign-language sentences remain, and about 1% of real Hinglish was probably filtered out.
- Test sets are not foreign-filtered, and test_b includes some Marathi-like and Gujarati-like text.
- Accented letters are missing from many source tweets. PHINC contains encoding damage (`â€¦`).
- Within train, only exact duplicates are removed.

## Licences
See `data/LICENSES.md`. HingCorpus is CC BY-NC-SA 4.0, so non-commercial and share-alike terms apply.
