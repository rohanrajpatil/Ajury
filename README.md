# silicon-jury

Research codebase for **measuring false-positive (wrongful-conviction) rates in
multi-agent LLM juries vs. human juries**, running on local open-source models
via Ollama. Scaffolding for the protocol in `mdd.md`.

The structure exists to keep the parts that *vary per experiment* (model, persona
mode, reasonable-doubt scaffolding) cleanly separated from the parts that don't
(deliberation mechanics, vote tallying, metrics). You change a hypothesis by
editing a config file, not by editing agent code.

## Layout

```
silicon-jury/
├── config/                       # declarative experiment arms
│   ├── default.yaml              # base config; every field maps to a protocol knob
│   └── experiments/*.yaml        # arms (extends default.yaml) for H1/H2/H3
├── data/
│   ├── raw/                      # source trial records (drop-in)
│   ├── processed/                # normalized {trial_id, transcript, metadata}
│   ├── personas/                 # generated OCEAN databank (JSON)
│   └── ground_truth/             # labels.json: actually_innocent per trial
├── src/silicon_jury/
│   ├── config.py                 # typed config + YAML `extends` inheritance
│   ├── llm/                      # ── BACKEND ABSTRACTION ──
│   │   ├── base.py               #   LLMClient ABC, Message, GenerationParams
│   │   ├── ollama_client.py      #   concrete local backend
│   │   ├── mock_client.py        #   no-network stub for tests
│   │   └── registry.py           #   logical name -> tag + size/quant (H3); factory
│   ├── personas/                 # ── COGNITIVE DIVERSITY ──
│   │   ├── models.py             #   OceanProfile, Persona (range-validated)
│   │   └── databank.py           #   generate / save / load / sample(12)
│   ├── prompts/                  # ── PROMPT ASSEMBLY ──
│   │   ├── *.txt                 #   editable templates (Blackstone, foreperson)
│   │   └── templates.py          #   fill templates; build task prompts
│   ├── agents/                   # ── ACTORS ──
│   │   ├── base.py               #   Agent: client + system prompt + memory
│   │   ├── juror.py              #   JurorAgent + verdict parsing
│   │   └── foreperson.py         #   neutral orchestrator + deterministic tally
│   ├── trial/                    # ── DOMAIN MODELS ──
│   │   ├── models.py             #   Trial, Ballot, Verdict, DeliberationResult
│   │   └── loader.py             #   load transcripts + merge ground truth (blinded)
│   ├── deliberation/             # ── WORKFLOW ──
│   │   ├── stage1_ballot.py      #   independent secret ballot
│   │   ├── stage2_debate.py      #   3-round structured debate + final vote
│   │   └── engine.py             #   assembles jury, runs stages, resolves verdict
│   ├── metrics/                  # ── ANALYSIS ──
│   │   ├── confusion.py          #   FPR/FNR (FP = wrongful conviction)
│   │   └── variance.py           #   consensus fraction + OCEAN↔guilt correlation
│   ├── experiment/
│   │   ├── runner.py             #   one config × trials × repetitions
│   │   └── sweep.py              #   multiple arms -> pooled records
│   └── reporting/report.py       #   per-arm FPR/FNR/drift summary table
├── scripts/                      # CLI entry points
└── tests/                        # pytest; runs without a GPU via MockClient
```

## How the abstractions isolate change

| Protocol hypothesis | What varies | Where it lives |
|---|---|---|
| H1 — groupthink in homogeneous juries | `jury.persona_mode` | `config` → `personas.databank.sample()` |
| H2 — anti-Bayesian drift without reasonable-doubt scaffolding | `deliberation.reasonable_doubt_scaffold` | `config` → `prompts.templates` |
| H3 — sensitivity to model size / quantization | `model.name` | `config` → `llm.registry` |

Nothing in `deliberation/`, `agents/`, or `metrics/` knows which hypothesis is
being tested. That is the point: the workflow is fixed, the inputs vary.

## Quickstart

```bash
pip install -r requirements.txt          # or: pip install -e .
ollama pull llama3:8b                     # local model

python scripts/generate_personas.py --n 120 --seed 42
# add processed trials to data/processed/ and labels to data/ground_truth/labels.json

python scripts/run_experiment.py \
    --config config/experiments/homogeneous_8b.yaml \
             config/experiments/heterogeneous_8b.yaml \
    --personas data/personas/databank.json \
    --out results/h1_summary.csv

pytest                                    # mock-backed, no GPU needed
```

## Key seams to extend

- **New backend** (cloud, llama.cpp): implement `LLMClient` in `llm/`, wire it in
  `registry.build_client`. Nothing else changes.
- **Different deliberation strategy** (e.g. ranked-choice, adversarial pairs):
  add a stage module and call it from `engine.run`; the runner and metrics are
  agnostic to stage internals.
- **Correlated OCEAN sampling**: replace `PersonaDatabank._draw_traits` with a
  multivariate sampler; everything downstream is unchanged.
- **Per-juror persona tracking in analysis**: `analyze_results.py` notes where to
  pair personas with ballots — currently the engine builds personas internally;
  expose them on `DeliberationResult` if you need that join.

## Important caveats (read before trusting any numbers)

- **OCEAN-as-prompt is a modeling assumption, not a validated mechanism.** Passing
  integer trait scores does not guarantee the model behaves like a person with
  those traits. Treat persona effects as "prompt-conditioning effects," not
  psychology, until you validate them.
- **Ground-truth "false positive" depends entirely on your labels.** Exoneration-
  registry cases are the defensible source; "standard convictions" are *not*
  confirmed true positives (some are undiscovered wrongful convictions), so FNR
  is noisier than FPR. Don't over-read FNR.
- **Transcripts drop demeanor, voir dire, and judicial instruction nuance.** The
  comparison to human juries is bounded by what survives in text.
- **Determinism is partial.** Seeds fix persona sampling and (if set) the model
  seed, but local inference can still vary. Use `run.repetitions` and report
  variance, not single runs.
