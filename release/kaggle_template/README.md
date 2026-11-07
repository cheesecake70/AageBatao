# HingMix v0.1: Roman-script Hinglish for next-word prediction

About 485k real Hinglish messages (Hindi and English mixed, written in Latin script), cleaned,
tagged word by word as Hindi or English, deduplicated, and split for language modelling. Built
for [AageBatao](https://github.com/AaditJain11/AageBatao), a next-word autocomplete for Hinglish
(live demo: https://aagebatao-965812381494.asia-south1.run.app).

This is a processed redistribution of three public corpora. Each file comes from exactly one
source and keeps that source's licence. Please cite the original authors (below).

## Files

| File | Rows | Tokens | Source | Genre | Licence |
| --- | --- | --- | --- | --- | --- |
| `hingcorpus_train.parquet` | 458,621 | 10,203,635 | L3Cube-HingCorpus | tweets | CC BY-NC-SA 4.0 |
| `hingcorpus_val.parquet` | 24,138 | 536,268 | L3Cube-HingCorpus | tweets | CC BY-NC-SA 4.0 |
| `cmu_dog_test.parquet` | 766 | 11,358 | CMU Hinglish DoG (test split) | chat | CC BY-SA 3.0 / GFDL |
| `phinc_test.parquet` | 1,160 | 19,379 | PHINC (seeded 10%) | social media | CC BY 4.0 |
| `vocab.json` | 30,000 | | top 30k train types | | CC BY-NC-SA 4.0 |

The two test sets are deliberately out of domain (chat and PHINC posts, not HingCorpus tweets).

## Columns

| Column | Meaning |
| --- | --- |
| `id` | `<source>_<hash>`, stable across rebuilds |
| `text` | normalised message |
| `tokens` | list of tokens |
| `lid` | language tag per token, aligned with `tokens`: `hi`, `en` or `other` (punctuation, emoji, `<url>`, `<user>`, `<num>`) |
| `n_switch` | number of hi↔en switches in the message |
| `cmi` | Code-Mixing Index (0 = one language, higher = more mixed) |
| `source` | `hingcorpus`, `cmu_dog` or `phinc` |
| `split` | `train`, `val`, `test_a`, `test_b` |
| `genre` | `tweet`, `chat` or `translation_pair` |

`vocab.json` is a list: index 0–3 are `<pad> <unk> <bos> <eos>`, then the 30k most frequent
train tokens.

## How it was built

1. Read the first 550,000 lines of the shuffled L3Cube-HingCorpus file, plus the CMU DoG test
   split and a seeded 10% of PHINC.
2. Normalise (URLs, mentions and numbers become `<url>`, `<user>`, `<num>`), tokenise, keep
   messages of 3 to 60 tokens.
3. Tag every token with HingBERT-LID; keep code-mixed messages; drop 10,488 foreign-language
   messages from train and val.
4. Remove exact duplicates, and train rows that near-duplicate a test row (MinHash).
5. Split. The full pipeline is `make dataset` in the GitHub repo.

## Known limitations

- The tagger is 99.6% accurate on HingLID gold but about 89% on a hand check of raw tweets.
  Short Hindi abbreviations (`h`, `b`, `kr`) are often tagged `en`, which creates false switches.
- Some foreign-language messages remain; about 1% of real Hinglish was probably filtered out.
- Test sets are not foreign-filtered; `phinc_test` has some Marathi-like and Gujarati-like text
  and encoding damage (`â€¦`).
- Tweets are informal and unmoderated, and may contain offensive language.

## Licence

Non-commercial use only, because most rows come from L3Cube-HingCorpus (CC BY-NC-SA 4.0). The
dataset as a whole is released under **CC BY-NC-SA 4.0**. `cmu_dog_test.parquet` stays under
CC BY-SA 3.0 / GFDL and `phinc_test.parquet` under CC BY 4.0, their original terms. Derivatives
must keep the same terms and credit the sources.

## Sources and credit

- **L3Cube-HingCorpus**: Nayak & Joshi, "L3Cube-HingCorpus and HingBERT: A Code Mixed
  Hindi-English Dataset and BERT Language Models", WILDRE-6 @ LREC 2022.
  https://github.com/l3cube-pune/code-mixed-nlp
- **CMU Hinglish DoG**: Hinglish translation of the CMU Document Grounded Conversations dataset
  (Zhou et al., EMNLP 2018). https://huggingface.co/datasets/festvox/cmu_hinglish_dog
- **PHINC**: Srivastava & Singh, "PHINC: A Parallel Hinglish Social Media Code-Mixed Corpus for
  Machine Translation", W-NUT @ EMNLP 2020. https://zenodo.org/record/3605597
- **HingBERT-LID** (language tags): L3Cube, same repository as HingCorpus.

Processed by Mokssha Nandu and Aadit Jain.
