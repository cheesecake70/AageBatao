# Phase 01 — Baselines and LSTM internals

Status: **done, acceptance met.** Numbers below are copied from files this repo generates; the file is
named next to each table. Nothing here is hand-estimated.

## Acceptance checklist (PRD §16)

- [x] KenLM 3/4/5-gram perplexities on `test_a` and `test_b`, bucketed
      → `reports/results/e0_ngram{3,4,5}_v0.1_real_test_{a,b}.csv`
- [x] Word LSTM size `m` trained on real v0.1, beats 5-gram by ≥ 15 % relative perplexity
      → `make summary VERSION=v0.1`: **246.14 vs 432.18 on test_a, +43.0 % (PASS)**
- [x] NumPy LSTM gradient check passes at 1e-6 on all parameters; loss curve on 50k sentences descends
      → `make grad-check` (all ≤ 3e-17); `reports/results/p1_numpy_loss.png` (7.87 → 5.74 nats)
- [x] `eval/report.py` emits the §9 template; buckets validated against hand-labelled examples
      → `tests/test_report.py`, `tests/test_buckets.py` (88 tests pass)

## 1. Setup

| | |
| --- | --- |
| dataset | hingmix v0.1: 458,621 train / 24,138 val / 766 test_a (cmu_dog) / 1,160 test_b (phinc) sentences; 10.2M train tokens |
| vocabulary | 30,000 types from train; `<unk>` rate overall test_a 8.39 %, test_b 13.29 % |
| hardware | n-gram + NumPy: Google Colab CPU. LSTMs: Kaggle, Tesla T4 16 GB, CUDA. Eval, bench, demo: Apple M2 CPU |
| seeds | 42 only (one seed; the 3-seed protocol is Part B) |
| run names | `e0_lstm_w_<size>_475k_real_s42` (the `475k` tag is the PRD's nominal size; actual rows 459k) |

Hidden-state policy in the trainer: **carried** across consecutive (shuffled) windows and detached
at every step. The windows are shuffled, so the carried state is a mild approximation; the PRD
accepts either carry or reset.

Training schedule for every size: AdamW, weight decay 0.01, linear warmup over the first 1,000
steps (of 1,357 per epoch) then cosine to zero at epoch 30, gradient clip 1.0, batch 128 × 64-token windows, early stop on
val perplexity with patience 3.

## 2. N-gram baselines (M0)

From `reports/results/p1_summary_v0.1.md`. Modified Kneser-Ney via KenLM `lmplz` with
`--discount_fallback`, no pruning, scored by renormalising KenLM's scores over our 30k vocabulary.

| order | test_a ppl | test_b ppl | test_a top-3 % | test_b top-3 % |
| --- | --- | --- | --- | --- |
| 3 | 431.27 | 340.27 | 21.62 | 28.97 |
| 4 | 431.85 | 341.74 | 21.67 | 29.07 |
| 5 | 432.18 | 341.61 | 21.67 | 29.15 |

Orders 3, 4 and 5 are within one perplexity point of each other on both test sets. With 10M tokens
of tweets almost no 4- or 5-gram context repeats, so the higher orders back off to the 3-gram
almost everywhere. This is the n-gram ceiling for this corpus; the bar for the LSTM is the 5-gram
on test_a, 432.18 × 0.85 = **367.35**.

## 3. Word LSTM (M1), sizes xs … xl

From `reports/results/p1_summary_v0.1.md` and `artifacts/e0_lstm_w_*_475k_real_s42_log.csv`.
Wall clock is total training time on the T4.

| size | params | non-emb | epochs | best val ppl | test_a ppl | test_b ppl | wall clock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xs | 2.0M | 0.06M | 30 | 232.96 | 293.13 | 231.93 | 57 min |
| s | 4.0M | 0.16M | 30 | 210.01 | 265.22 | 216.54 | 76 min |
| m | 8.8M | 1.08M | 30 | 196.72 | 246.14 | 204.61 | 118 min |
| l | 19.6M | 4.23M | 30 | 168.98 | 224.72 | 192.82 | 233 min |
| xl | 60.8M | 14.73M | 17 (early stop, best 14) | 167.78 | 242.38 | 200.94 | 311 min |

**Acceptance:** `m` vs 5-gram on test_a: 246.14 vs 432.18 = **43.0 % relative perplexity** (need ≥ 15).
Top-3 accuracy on test_a goes from 21.7 % (5-gram) to 30.4 % (m) and 31.4 % (l); top-1 from 8.8 % to
16.4 %.

Sizes xs to l never early-stopped: val perplexity was still improving at epoch 30, by well under
one point per epoch, because the cosine schedule had driven the learning rate to zero. The schedule
ended those runs, not the data. `xl` is the only run that stopped on patience.

Figures: `reports/figures/figA_scaling_v0.1.png`, `figB_buckets_test_a_v0.1.png`,
`figB_buckets_test_b_v0.1.png`, `figC_curves_v0.1.png`.

### Bucketed, `m` vs 5-gram (test_a)

| bucket | share % | 5-gram ppl | LSTM m ppl | 5-gram top-3 % | LSTM m top-3 % |
| --- | --- | --- | --- | --- | --- |
| overall | 100.0 | 432.2 | 246.1 | 21.7 | 30.4 |
| hi | 43.0 | 407.8 | 228.6 | 16.7 | 24.5 |
| en | 12.7 | 1288.6 | 1158.1 | 19.4 | 22.6 |
| switch hi→en | 14.3 | 3931.3 | 2575.2 | 12.1 | 15.3 |
| switch en→hi | 13.7 | 663.0 | 173.6 | 7.4 | 21.9 |
| other | 16.3 | 21.5 | 15.2 | 57.0 | 72.4 |

## 4. NumPy LSTM (M4)

`make grad-check` output on this M2 (hand-derived BPTT vs `torch.nn.LSTM` + autograd, same
weights, same 8-token sequence; tolerance 1e-6):

```
emb   max|Δ| = 1.73e-18  OK
W     max|Δ| = 1.73e-18  OK
b     max|Δ| = 1.39e-17  OK
Wout  max|Δ| = 2.17e-18  OK
bout  max|Δ| = 2.78e-17  OK
loss  max|Δ| = 4.44e-16  OK
```

Loss curve on 50,000 real sentences at H = 64, own 5,000-word vocabulary, plain SGD with lr 0.5 and
clip 1.0, one 32-token window per step, 5,000 steps, 93 s on a Colab CPU:
`reports/results/p1_numpy_loss.png`. First-50 vs last-50 step mean **7.867 → 5.736** nats, uniform =
ln V = 8.517. The curve drops fast to about 6.0 by step 1,000 and then creeps, which is what a
64-unit LSTM on single windows with no momentum should do; the point of the run is that the
hand-written backward learns, not the number it reaches.

What deriving dL/dW_f by hand taught us. The forget gate's gradient is the sum over time of
δc_t ⊙ c_{t-1} ⊙ f_t ⊙ (1 − f_t) pushed through the stacked input [x_t; h_{t-1}], and δc_t itself is
δh_t ⊙ o_t ⊙ (1 − tanh²(c_t)) plus the gradient flowing in from c_{t+1} scaled by f_{t+1}. Writing
that second term out is the whole reason an LSTM trains at depth: the cell-state path is multiplied
by f_{t+1} at each step instead of by a weight matrix, so with forget bias +1 the gradient reaches
back many tokens without vanishing. Two things only became obvious by implementing them: the four
gates share one [E+H, 4H] matrix so their gradients are one outer product per step, not four; and
dh_{t-1} receives contributions from all four gates at step t plus the output-layer gradient at
step t−1, so the order of accumulation inside the backward loop matters and a single missed term
shows up as a 1e-2 error in the check, never as a subtle one.

## 5. Observations

- **Switch points.** *(smoke 3-gram: switch hi→en ppl ≈ 23k vs hi ≈ 490; the LSTM should move
  this bucket most.)* The LSTM's largest single gain is the opposite switch. On test_a, en→hi goes
  from 663 (5-gram) to 174 (m) and 157 (l), and its top-3 from 7.4 % to 21.9 %: once the model has
  a few English words of context it predicts the Hindi continuation almost as well as plain Hindi
  (229). hi→en moves less, 3931 → 2575 (m), and stays the hardest bucket for every model by an
  order of magnitude. Predicting the first English word after Hindi is a vocabulary problem as much
  as a context problem: that bucket's `<unk>` rate is 16 %, and the words that do exist in the vocab
  are spread thin. Two LSTM improvements go the same way: more model (l) helps hi→en a little, more
  data would help it more.

- **`<unk>` per bucket.** *(smoke: English OOV 23 % vs Hindi 7.7 % on test_a — the opposite of
  the PRD's expectation, likely cmu_dog movie names vs a hingcorpus-only vocab. Does it hold at
  475k?)* It holds. On test_a the English bucket is 18.8 % `<unk>` against 7.0 % for Hindi. The
  training vocabulary is built from hingcorpus tweets, which are Hindi-heavy and carry a narrow
  English register, while cmu_dog is English-leaning movie chat with named entities. On test_b
  (phinc, social) the gap closes to 12.9 % vs 12.3 %, so the asymmetry is a domain effect of the
  test set, not a property of Hinglish. English perplexity is also where the LSTM gains least,
  1289 → 1158 (m) → 1071 (l): a model cannot predict a word it does not have. The §3 PRD claim that
  Hindi will have the higher OOV rate is wrong for this corpus and should be rewritten.

- **Scaling.** Fig A is a clean log-linear descent from xs to l on both test sets, about 20
  perplexity points per 4× non-embedding parameters, and then bends up at xl. xl reaches the best
  validation perplexity of any size (167.78 vs 168.98 for l) but is worse on both test sets (242 vs
  225 on test_a, 201 vs 193 on test_b). Fig C shows why: xl's train perplexity at its best epoch is
  129 against val 168, a 30 % gap, where l sits at 186 vs 169. xl is the only untied model, with a
  15M-parameter output layer, and 10M tokens is not enough for it; it memorises tweets and the
  in-distribution val set rewards that slightly while out-of-domain test sets punish it. The
  practical size for this data is l. A bigger model is not the next step; more data is, which is
  the P2 synthetic-augmentation question.

- **test_a vs test_b.** test_b (phinc, social) is easier overall for every model, 205 vs 246 for m,
  but only because 29 % of its tokens are the `other` bucket (punctuation, numbers, emoji) at
  perplexity 26. On the language buckets test_b is harder: Hindi 338 vs 229, English 965 vs 1158
  the other way, and `<unk>` 13.3 % overall vs 8.4 %. Chat (test_a) has long Hindi runs that the
  LSTM handles well and a hard hi→en switch; social (test_b) has short, noisy, high-OOV text where
  the switch buckets are closer to the plain-language ones. Overall perplexity on these two sets is
  not comparable; the per-bucket rows are.

## 6. Demo

`make demo CKPT=artifacts/e0_lstm_w_m_475k_real_s42_best.pt SIZE=m VERSION=v0.1` serves the
Tab-to-accept page on the `m` checkpoint. The same image is deployed on Google Cloud Run at
https://aagebatao-965812381494.asia-south1.run.app (asia-south1, 1 CPU, 1 GiB, scale-to-zero;
`make deploy`). Measured from this laptop: first request after a cold start 0.5 s, then ~30 ms
model time and ~75 ms round trip. After "kya kar raha" the ghost is "hai" at 58 %; the smoke
model offered the same word at 0.7 %. Latency from `make bench`
(`reports/results/p1_latency_v0.1.json`, 200 prefixes, in-process on an M2 CPU): p50 **2.61 ms**,
p95 **3.66 ms**, max 6.08 ms, target p95 ≤ 30 ms. Met with an order of magnitude to spare; the `l`
model would also fit.

Known demo behaviours, left as-is for P1: punctuation and emoji are vocabulary items and are
suggested when the corpus says so; the page shows the top suggestion regardless of its probability,
so after one word of context the ghost is often a guess. A confidence threshold is a one-line
follow-up once there is a product reason to tune it.

## 7. Open questions for Part B

1. **Seeds.** Everything above is seed 42. The xl-vs-l ordering on test sets (17 points) and the
   m-vs-5-gram margin (186 points) are both far larger than typical seed noise for this setup, but
   the PRD's 3-seed protocol has not been run. Two more `m` seeds (13, 2024) are about 4 GPU hours.
2. **Why does xl overfit and where is the elbow?** Tied vs untied is confounded with size: xl is
   the only untied model. A tied xl, or an untied l, separates the two. Also worth a dropout sweep on
   l, which still had a 10 % train/val gap at epoch 30.
3. **Is the cosine horizon too short?** xs through l were still improving at epoch 30 with lr ≈ 0.
   A 40-epoch run on l costs 5 hours and tells us whether the l number is schedule-limited.
4. **English coverage.** The vocabulary is the ceiling on the `en` and `hi→en` buckets. Options:
   a larger vocab, a character or subword fallback, or augmentation that adds English register.
   This is the first thing P2's synthetic data should be measured on, per bucket, not overall.
5. **The PRD's `<unk>` expectation (§3, §9)** should be corrected to say that English OOV is higher
   on chat-domain test sets with a tweet-built vocabulary.
6. **Evaluation cost.** Scoring the n-gram over the full 30k vocabulary per position takes ~11 min
   per order per test set in pure Python. Fine for P1, too slow for a sweep; vectorising the KenLM
   query or caching per-context rows is the fix if Part B re-scores n-grams often.
