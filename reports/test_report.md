# Test report

Generated 2026-11-06 14:56 by `make test-report`. Do not edit; rerun instead.

## Summary

| | |
| --- | --- |
| tests | 88 |
| passed | 88 |
| failed | 0 |
| skipped | 0 |
| wall time | 15.0 s |
| python | 3.11.15 |
| `ruff check src tests scripts` | clean |
| `black --check src tests scripts` | clean |

**Verdict: ALL PASS.**

## By file

| file | tests | passed | time |
| --- | --- | --- | --- |
| `test_app.py` | 5 | 5 | 0.14 s |
| `test_bench.py` | 1 | 1 | 0.42 s |
| `test_buckets.py` | 8 | 8 | 0.01 s |
| `test_dataset.py` | 7 | 7 | 0.02 s |
| `test_dedupe.py` | 2 | 2 | 0.03 s |
| `test_filter.py` | 3 | 3 | 0.00 s |
| `test_foreign.py` | 2 | 2 | 0.00 s |
| `test_lstm.py` | 5 | 5 | 0.03 s |
| `test_lstm_numpy.py` | 8 | 8 | 0.53 s |
| `test_metrics.py` | 4 | 4 | 0.01 s |
| `test_naming.py` | 2 | 2 | 0.00 s |
| `test_ngram.py` | 3 | 3 | 1.98 s |
| `test_normalize.py` | 4 | 4 | 0.01 s |
| `test_predictor.py` | 7 | 7 | 0.09 s |
| `test_report.py` | 7 | 7 | 0.07 s |
| `test_summarize.py` | 5 | 5 | 2.72 s |
| `test_tokenize.py` | 9 | 9 | 0.01 s |
| `test_trainer.py` | 6 | 6 | 8.96 s |

## Every test

### `test_app.py`

FastAPI routes /, /health, POST /predict and input validation

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_index_serves_page` | index serves page | pass | 0.072 s |
| `test_health` | health | pass | 0.018 s |
| `test_predict_endpoint` | predict endpoint | pass | 0.025 s |
| `test_predict_partial_and_validation` | predict partial and validation | pass | 0.021 s |
| `test_app_without_predictor_needs_env` | app without predictor needs env | pass | 0.009 s |

### `test_bench.py`

Latency bench: prefix sampling and p50/p95 stats

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_sample_prefixes_are_real_prefixes` | sample prefixes are real prefixes | pass | 0.425 s |

### `test_buckets.py`

P1 §9 bucket assignment: hi / en / switch hi→en / switch en→hi / other

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_assign_hand_labelled[lid0-expected0]` | assign hand labelled | pass | 0.001 s |
| `test_assign_hand_labelled[lid1-expected1]` | assign hand labelled | pass | 0.002 s |
| `test_assign_hand_labelled[lid2-expected2]` | assign hand labelled | pass | 0.001 s |
| `test_assign_hand_labelled[lid3-expected3]` | assign hand labelled | pass | 0.001 s |
| `test_assign_hand_labelled[lid4-expected4]` | assign hand labelled | pass | 0.001 s |
| `test_assign_rejects_unknown_label` | assign rejects unknown label | pass | 0.001 s |
| `test_target_buckets_appends_eos_as_other` | target buckets appends eos as other | pass | 0.001 s |
| `test_bucket_names_match_prd_section_14` | bucket names match prd section 14 | pass | 0.001 s |

### `test_dataset.py`

P1 windows: 64-token teacher-forced x/y shift, loaders, eval sentences

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_encode_sentences_wraps_with_bos_eos_and_unk` | encode sentences wraps with bos eos and unk | pass | 0.002 s |
| `test_window_dataset_shapes_and_shift` | window dataset shapes and shift | pass | 0.001 s |
| `test_window_dataset_exact_fit_needs_one_extra_token` | window dataset exact fit needs one extra token | pass | 0.001 s |
| `test_make_loader_fixed_batch_shape` | make loader fixed batch shape | pass | 0.003 s |
| `test_eval_sentences_keeps_lid_aligned` | eval sentences keeps lid aligned | pass | 0.003 s |
| `test_load_split_reads_parquet` | load split reads parquet | pass | 0.009 s |
| `test_load_split_missing_file_is_helpful` | load split missing file is helpful | pass | 0.002 s |

### `test_dedupe.py`

P0 near-duplicate removal (MinHash) in the dataset build

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_jaccard_identical` | jaccard identical | pass | 0.001 s |
| `test_index_finds_near_duplicate_case_insensitive` | index finds near duplicate case insensitive | pass | 0.030 s |

### `test_filter.py`

P0 sentence filters (length, script, junk) in the dataset build

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_n_switch_skips_other` | n switch skips other | pass | 0.001 s |
| `test_cmi` | cmi | pass | 0.001 s |
| `test_code_mixed` | code mixed | pass | 0.001 s |

### `test_foreign.py`

P0 foreign-language sentence removal in the dataset build

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_coverage_ignores_punctuation_and_specials` | coverage ignores punctuation and specials | pass | 0.001 s |
| `test_foreign_sentence_flagged` | foreign sentence flagged | pass | 0.001 s |

### `test_lstm.py`

M1 torch word LSTM: forward, tied weights, parameter counts, log_probs

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_forward_shapes_and_state` | forward shapes and state | pass | 0.010 s |
| `test_tied_weights_share_storage` | tied weights share storage | pass | 0.006 s |
| `test_log_probs_interface` | log probs interface | pass | 0.005 s |
| `test_count_params` | count params | pass | 0.004 s |
| `test_from_config_matches_yaml_keys` | from config matches yaml keys | pass | 0.005 s |

### `test_lstm_numpy.py`

M4 NumPy LSTM: shapes, loss, hand-written BPTT vs torch

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_param_shapes` | param shapes | pass | 0.003 s |
| `test_forward_shapes_and_loss` | forward shapes and loss | pass | 0.002 s |
| `test_helpers` | helpers | pass | 0.001 s |
| `test_backward_shapes` | backward shapes | pass | 0.003 s |
| `test_gradients_match_torch` | P1 checkbox 3, first half: every parameter gradient within 1e-6 of torch.nn.LSTM. | pass | 0.183 s |
| `test_gradients_match_torch_with_repeated_ids` | gradients match torch with repeated ids | pass | 0.009 s |
| `test_sgd_step_clips_and_moves` | sgd step clips and moves | pass | 0.004 s |
| `test_loss_descends_on_tiny_data` | loss descends on tiny data | pass | 0.326 s |

### `test_metrics.py`

P1 perplexity, top-k rank and <unk> rate per bucket

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_score_sentence_nll_rank_unk` | score sentence nll rank unk | pass | 0.003 s |
| `test_score_sentence_shape_mismatch` | score sentence shape mismatch | pass | 0.001 s |
| `test_aggregate_per_bucket_values` | aggregate per bucket values | pass | 0.001 s |
| `test_aggregate_empty_bucket_is_nan_with_zero_share` | aggregate empty bucket is nan with zero share | pass | 0.001 s |

### `test_naming.py`

Run naming convention e<exp>_<family>_<size>_<data>_<mix>_s<seed>

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_run_name_matches_prd_example` | run name matches prd example | pass | 0.001 s |
| `test_run_name_rejects_unknown_seed` | run name rejects unknown seed | pass | 0.001 s |

### `test_ngram.py`

M0 KenLM wrapper: training text, log-prob interface, renormalisation

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_write_train_text_substitutes_unk` | write train text substitutes unk | pass | 0.005 s |
| `test_log10_constant` | log10 constant | pass | 0.002 s |
| `test_ngram_end_to_end` | ngram end to end | pass | 1.977 s |

### `test_normalize.py`

P0 text normalisation (NFC, URLs, mentions, numbers, emoji)

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_replaces_entities` | replaces entities | pass | 0.002 s |
| `test_preserves_case_and_spelling` | preserves case and spelling | pass | 0.001 s |
| `test_nfc_and_emoji` | nfc and emoji | pass | 0.002 s |
| `test_digits_inside_words_stay` | digits inside words stay | pass | 0.001 s |

### `test_predictor.py`

Demo predictor: partial-word filtering, blocked specials, latency

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_split_input_detects_partial_word` | split input detects partial word | pass | 0.001 s |
| `test_split_input_normalises_like_training` | split input normalises like training | pass | 0.001 s |
| `test_predict_returns_k_real_words` | predict returns k real words | pass | 0.026 s |
| `test_predict_filters_by_typed_prefix` | predict filters by typed prefix | pass | 0.009 s |
| `test_predict_prefix_is_case_insensitive_and_skips_exact` | predict prefix is case insensitive and skips exact | pass | 0.005 s |
| `test_predict_empty_text_gives_first_word` | predict empty text gives first word | pass | 0.009 s |
| `test_load_from_checkpoint` | load from checkpoint | pass | 0.037 s |

### `test_report.py`

P1 §9 markdown/csv result tables

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_uniform_perplexity_equals_vocab_size` | uniform perplexity equals vocab size | pass | 0.006 s |
| `test_uniform_topk_is_all_ties` | uniform topk is all ties | pass | 0.002 s |
| `test_unk_rate_and_shares` | unk rate and shares | pass | 0.002 s |
| `test_markdown_table_shape` | markdown table shape | pass | 0.002 s |
| `test_markdown_empty_bucket_renders_dash` | markdown empty bucket renders dash | pass | 0.001 s |
| `test_dataframe_and_csv` | dataframe and csv | pass | 0.055 s |
| `test_evaluate_model_rejects_misaligned_lid` | evaluate model rejects misaligned lid | pass | 0.002 s |

### `test_summarize.py`

Phase summary: result collection, wide table, acceptance verdict

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_model_label_parses_both_naming_schemes` | model label parses both naming schemes | pass | 0.002 s |
| `test_collect_wide_and_acceptance` | collect wide and acceptance | pass | 0.080 s |
| `test_acceptance_fail_and_missing` | acceptance fail and missing | pass | 0.054 s |
| `test_collect_raises_when_nothing_matches` | collect raises when nothing matches | pass | 0.004 s |
| `test_figures_render` | figures render | pass | 2.579 s |

### `test_tokenize.py`

P0 tokeniser and 30k vocabulary (specials, encode/decode, save/load)

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_punctuation_and_specials` | punctuation and specials | pass | 0.001 s |
| `test_emoji_is_single_token` | emoji is single token | pass | 0.001 s |
| `test_specials_have_fixed_ids` | specials have fixed ids | pass | 0.001 s |
| `test_build_orders_by_frequency_then_alphabet` | build orders by frequency then alphabet | pass | 0.001 s |
| `test_build_caps_size_including_specials` | build caps size including specials | pass | 0.001 s |
| `test_encode_maps_unknown_to_unk_and_round_trips` | encode maps unknown to unk and round trips | pass | 0.001 s |
| `test_save_load_round_trip` | save load round trip | pass | 0.003 s |
| `test_constructor_rejects_missing_specials` | constructor rejects missing specials | pass | 0.001 s |
| `test_build_is_deterministic` | build is deterministic | pass | 0.002 s |

### `test_trainer.py`

Trainer: schedule, clipping, early stop, checkpoint and csv log

| test | checks | result | time |
| --- | --- | --- | --- |
| `test_seed_everything_is_reproducible` | seed everything is reproducible | pass | 0.009 s |
| `test_scheduler_warmup_then_cosine` | scheduler warmup then cosine | pass | 8.599 s |
| `test_scheduler_zero_warmup` | scheduler zero warmup | pass | 0.001 s |
| `test_fit_learns_saves_and_logs` | fit learns saves and logs | pass | 0.243 s |
| `test_early_stopping_triggers` | early stopping triggers | pass | 0.103 s |
| `test_detach_cuts_graph` | detach cuts graph | pass | 0.002 s |
