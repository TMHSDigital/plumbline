# Example report

This is real, unedited output from `plumbline run`. Everything below the rule is
exactly what the tool printed. Nothing in it was tuned to look good.

| | |
|---|---|
| Dataset | `datasets/public/jevbench-hard.jsonl`, the vendored JevBench public fixture |
| Rows | 111 read, 105 scored, 6 held back as ordinal score rows |
| Arm | the `mock` adapter, seed 7, target accuracy 0.8 |
| Date | 2026-09-21 |
| Command | `plumbline run datasets/public/jevbench-hard.jsonl --adapter mock --format jevbench --seed 7 --accuracy 0.8 --report example.md` |

**The arm is a seeded mock, not a vendor.** It is a deterministic stand-in that
draws answers at a target accuracy, and its numbers are properties of plumbline's
own harness rather than measurements of anybody's model. It is used here for two
reasons: it makes this page reproducible by anyone who clones the repository and
runs one command, and the point of the page is what the tool refuses to conclude,
which does not depend on who answered.

What to look at, in order:

- **ECE is 0.0740 and the floor is 0.0707.** The tool reports the figure and then
  calls it inconclusive: a perfectly calibrated model would often score this
  badly on 105 rows, so these rows cannot tell the two apart. Read that as the
  absence of a result, not as a pass. A tool that printed 0.0740 on its own
  would be handing you a number that looks like a finding and is not one. Every
  inferential figure on the page is read against its own null this way; cost and
  latency are measurements rather than inferences and have no null.
- **Cost says why it is blank rather than showing a zero.** The basis is "the
  model that answered is not priced", which is a different fact from "this
  adapter cannot report tokens". A zero in that column would read as free.
- **Recalibration refuses.** It needs 200 held-out rows and this split has 53.
  It declines to emit a temperature rather than fitting one whose uncertainty
  exceeds the correction it claims to make.
- **The cascade refuses too,** because a threshold depends on what an escalation
  and an error cost you, and no benchmark can know those.
- **The noul rows are flagged.** 38 yes/no rows were asked as two-option choice
  questions, which is a different question from the one the dataset states, and
  the report says so instead of averaging across the difference.

Running the same command against a real vendor produces the same shape. See
`docs/PLAN.md` for what a live run measured.

---

# plumbline report

Generated 2026-09-22 against dataset `c18e9496`, 105 rows. 1 arm(s).

## How to read this

- Every figure states the rows it was computed on and the null it is read against. A number on its own is not a finding.
- **"INCONCLUSIVE" is not a pass.** It means the value sits inside what the null already produces at this sample size, so this dataset cannot tell the two apart. Nothing was established in either direction. A model that is genuinely well calibrated and one that is badly calibrated can both land here on too few rows, and the figure does not say which you have.
- Arms are grouped by what kind of number they report. **Figures in different groups are not comparable** and are never placed side by side.
- "Not reported" is a result with a reason attached, not a missing cell.

## Dataset

- 111 rows read from datasets\public\jevbench-hard.jsonl, 111 loaded, 0 refused. Translated from JevBench: the case text is the row's question above its state, and every row is asked as a one-of-n choice. plumbline's harness, prompts and scoring differ from JevBench's, so these numbers are not comparable with theirs. 6 rows wrote the gold label as a JSON number against string options; each was matched to the option of the same name. 38 rows carried criteria that do not describe the options one for one, so their option descriptions were dropped rather than guessed. 6 rows ask for an ordinal score. plumbline v0.1 has no ordinal support -- flattening levels into unordered options discards the ordering -- so they are loaded, marked, and excluded from scored results.
- 6 score rows are excluded from every figure below: plumbline v0.1 scores choice and yes/no questions only.


---

## Calibrated claims

The vendor asserts these probabilities are calibrated. Whether that survives contact with this dataset is what the figures below answer.

### mock

- **Model**: requested `mock-1`, reported `mock-1`.
- **Asked**: 67 choice asked as choice, 38 noul asked as choice.
  - 38 noul rows were asked as choice questions, which is a different question from the one the dataset states. Not comparable with an arm that asked them as noul.
- Accuracy 0.7714 over 105 rows, against a chance null of 0.3416 (95th percentile 0.4190): better than chance at this sample size.
- ECE 0.0740 over 105 rows (10 equal width bins), against a calibrated-model floor of 0.0707 (95th percentile 0.1109): INCONCLUSIVE at this sample size. A perfectly calibrated model would often score this badly on this many rows, so this dataset cannot tell the two apart. This is not a clean bill of health: nothing was established either way. Collect more rows to make the question answerable.
- Brier 0.1711 over 105 rows, against a calibrated-model floor of 0.1489 (95th percentile 0.1839): INCONCLUSIVE at this sample size. A perfectly calibrated model would often score this badly on this many rows, so this dataset cannot tell the two apart. This is not a clean bill of health: nothing was established either way. Collect more rows to make the question answerable.
- **Confidence**: AUROC 0.6034 over 105 rows, against a permutation null of 0.4997 (95th percentile 0.6116): INCONCLUSIVE at this sample size. Permutation would often score this well on this many rows, so this dataset cannot tell the two apart. This is not a result in either direction. Collect more rows to make the question answerable.
- **Cost**: not reported. No cost available. None of the 105 cases could be priced, so cost is not reported rather than being shown as zero. 105 rows: tokens were reported, but the model that answered is not priced.
- **Latency**: p50 37.7ms, p95 85.0ms, p99 104.7ms over 105 live calls

#### Recalibration

- Not reported. recalibration needs at least 200 held-out evaluation rows and this split has 53. Fitting a temperature on fewer rows produces a number whose uncertainty is larger than the correction it claims to make, and it arrives looking like a measurement. Collect more labeled rows, or skip recalibration and report raw calibration only.

#### Cascade

- Not reported. A threshold is decided by two numbers no benchmark can know: what one escalation to the expensive arm costs, and what one wrong answer costs. Supply both (`--escalation-cost`, `--error-cost`) and this section states where to set the threshold and what it buys.

#### Diagnostics

Read these only after the figures above. MCE is a maximum over bins, decided by one bin, and at a few hundred rows it cannot detect overconfidence spread evenly across the range.

- MCE 0.1588 over 105 rows (10 equal width bins), against a calibrated-model floor of 0.1251 (95th percentile 0.2198): INCONCLUSIVE at this sample size. A perfectly calibrated model would often score this badly on this many rows, so this dataset cannot tell the two apart. This is not a clean bill of health: nothing was established either way. Collect more rows to make the question answerable.
- Multiclass Brier 0.3526 over 105 rows, against a calibrated-model floor of 0.3168 (95th percentile 0.3975): INCONCLUSIVE at this sample size. A perfectly calibrated model would often score this badly on this many rows, so this dataset cannot tell the two apart. This is not a clean bill of health: nothing was established either way. Collect more rows to make the question answerable.
