# Your own data

plumbline measures a model on your labeled rows, so this is the file format
you write them in, and the commands you run next. Put your file anywhere; the
repository's [`datasets/private/`](../datasets/private) directory is gitignored
for exactly this, so a clone never commits it.

## The format

One JSON object per line (JSONL), UTF-8, with or without a byte order mark.
Blank lines are skipped. Every other line is one case.

| field | required | what it is |
|---|---|---|
| `id` | yes | A non-empty string, unique in the file. It names the row in the artifact. |
| `text` | yes | A non-empty string: what the model is shown. |
| `labels` | yes | A list of at least two distinct strings: the options it chooses between. |
| `gold_label` | yes | The correct option. It must be one of `labels`. |
| `question_type` | no | `choice` (the default), `noul`, or `score`. See below. |
| `label_descriptions` | no | An object mapping an option to a sentence describing it. `typesafe_wire` sends them as the choice's criteria; the other adapters do not use them, and the report says so when the rows carried any. |

`question_type` says what the row actually asks, and plumbline asks it that way:

- `choice`: pick one of the options. Most classification rows are this.
- `noul`: a yes/no question. The options must be a yes/no pair (`yes`/`no` or
  `true`/`false`), because which option is the yes is a fact about your data,
  not something to guess. A transport that has a native yes/no question asks it
  as one, and gets back one probability rather than a distribution.
- `score`: an ordinal level. Its `labels` are its levels, the integers 0 to
  K minus 1 as strings, and a row whose options are not is refused. It is read
  by rank, by three figures of its own, and never by the choice figures, which
  would score wrong by one level and wrong by three the same. A level's
  `label_descriptions` entry is its rubric, which a score transport sends.

## An example

```jsonl
{"id": "t-1", "text": "My card was charged twice for one order.", "labels": ["billing", "shipping", "returns", "other"], "gold_label": "billing"}
{"id": "t-2", "text": "The parcel says delivered but it is not here.", "labels": ["billing", "shipping", "returns", "other"], "gold_label": "shipping"}
{"id": "t-3", "text": "Is this order eligible for a refund?", "labels": ["yes", "no"], "gold_label": "yes", "question_type": "noul"}
{"id": "t-4", "text": "How urgent is this complaint, 0 to 3?", "labels": ["0", "1", "2", "3"], "gold_label": "2", "question_type": "score"}
{"id": "t-5", "text": "Where is my refund?", "labels": ["billing", "returns"], "gold_label": "returns", "label_descriptions": {"billing": "charges and invoices", "returns": "sending an item back, and its refund"}}
```

## What is refused, and why

A row that cannot be scored honestly is refused with its line number, and the
rest of the file still runs. `--strict` refuses the whole run instead.

- A missing required field, an empty `id` or `text`, or `labels` that is not a
  list of at least two distinct, non-empty strings. A label written as `null`,
  `true` or a number is refused rather than turned into the text `"None"`,
  `"True"` or `"1"`.
- A `label_descriptions` entry for something that is not one of the row's
  options, or a description that is not a non-empty string.
- A `gold_label` that is not one of the row's own `labels`. Scoring it would
  mark every model wrong on that row, which reads as a model failure and is a
  data error.
- A `question_type` other than the three above. An unknown type is not assumed
  to be a choice.
- An `id` already used earlier in the file.
- A line that is not a JSON object.

A file with no rows at all, a path that does not exist, or a file that is not
UTF-8 is an error rather than an empty run; the last names the line with the
byte that is not.

## Running it

The mock needs nothing and spends nothing, so start there to check the file:

```
uv run plumbline run my-data.jsonl --adapter mock --results results --report results/report.md
```

A hosted vendor needs its key in the environment and, for a cost column, a
pricing table you filled in (see the README). `--limit` runs only the first N
cases; `--max-cases` and `--max-cost-usd` refuse the run instead of truncating
it, so they are guards rather than ways to run less:

```
uv run plumbline run my-data.jsonl --adapter typesafe_wire --model jev-latest \
    --limit 40 --max-cost-usd 0.50 --pricing my-pricing.json \
    --results results --report results/report.md
```

A local checkpoint needs the `local` extra (`uv sync --extra local`), a model
id, and the commit it is pinned to, as 40 hex characters. A branch name or a tag
can move, and everything measured would then be attributed to other weights:

```
uv run plumbline run my-data.jsonl --adapter local_logits \
    --model Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
    --results results --report results/report.md
```

## The cascade needs two numbers only you know

Where to set a threshold depends on what one escalation to the expensive path
costs you and what one wrong answer costs you. plumbline never defaults either,
because a made-up default would decide the threshold. Pass both, in USD:

```
uv run plumbline run my-data.jsonl --adapter mock --escalation-cost 0.02 --error-cost 1.00 \
    --results results --report results/report.md
```

The threshold is chosen on half of the scored rows and judged on the other half,
so a threshold needs at least 400 scored rows; below that the report says so
instead of printing one.

## Reports from runs that already happened

Every run writes an artifact to `--results`. `plumbline report` renders one
document from one or more of them, so a finished run is never repeated to get a
report, or to try different cascade costs:

```
uv run plumbline report results/*.json --out results/report.md \
    --escalation-cost 0.02 --error-cost 1.00
```
