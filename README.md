# Moral foundations post-training

Post-train an open LLM to **value** the six MFQ-2 moral foundations (Care, Equality,
Proportionality, Loyalty, Authority, Purity; Atari et al. 2023) the way a real group of people
does. The model's first token in every reply is the foundation that wins for that profile (or
`None` for non-moral prompts). This "moral reflex" is trained with SFT and then DPO, one LoRA
adapter per profile.

## Data: generated for training, human for testing

| Role | Data | Human? |
|---|---|---|
| **Target** | Profiles = mean MFQ-2 scores of real survey groups (`configs/profiles.yaml`): US liberals (n=304) and conservatives (n=412) from Study 3, Japan and Egypt from Study 2 | yes |
| **Train** | ~1,000 generated dilemmas where two foundations conflict, each with one response per side; the profile picks the winner | no |
| **Test** | MFQ-2's 36 items, scored against the target group's real scores | yes |
| **Test** | 132 Clifford et al. (2015) vignettes with human wrongness ratings | yes |
| **Test** | Held-out generated dilemmas (10% split + the whole "AI assistant with tools" domain) | no |

Neither the questionnaire items nor the vignettes ever appear in training. Sources and licences
are in `data/raw/SOURCES.md`. Findings about the data and models are logged in `FINDINGS.md`.

## Steps

```bash
# Local: no GPU needed
python -m mft.profiles from-data                      # recompute profiles from raw survey data
python -m mft.generate_dilemmas --limit-jobs 2        # smoke test (provider/model from .env)
python -m mft.generate_dilemmas                       # full run -> data/generated/dilemmas.jsonl
python -m mft.verify_dilemmas --llm-model gpt-5.4-mini  # blind label check -> dilemmas.verified.jsonl
python -m mft.decontaminate --threshold <t>          # drop near-duplicates of human test items -> dilemmas.clean.jsonl

```

## Cloud GPU on AWS (one stack, one-command teardown)

```bash
scripts/aws/gpu.sh up                  # g6e.xlarge (L40S 48GB, ~$1.86/hr); auto-terminates after MAX_HOURS=6
scripts/aws/gpu.sh push                # code + data + deps
QUICK=1 scripts/aws/gpu.sh run us_liberal   # fast end-to-end check first
scripts/aws/gpu.sh logs
scripts/aws/gpu.sh run us_liberal      # then the real runs (one at a time; tmux session 'mft')
scripts/aws/gpu.sh status              # uptime and cost so far
scripts/aws/gpu.sh down                # pulls results/ + runs/, deletes everything, confirms 0 instances left
```

Requires the EC2 quota "Running On-Demand G and VT instances" >= 4 vCPUs (new accounts have 0).

`.env` settings: `OPENAI_API_KEY` and `OPENAI_MODEL` (the default provider), or `MFT_PROVIDER=anthropic`
with `ANTHROPIC_API_KEY`.

## Evaluations (`python -m mft.evaluate ...`)

- `first-token`: does the reflex pick the profile's side? (KL to soft target, accuracy)
- `forced-tag`: does the tag steer what follows? (~0 means the tag is decorative)
- `mfq2`: does the self-reported profile move toward the real group? (MAE, Pearson)
- `vignettes`: do wrongness ratings of real stimuli follow the profile's ranking?
- `behavior`: in free generation with no tag forced, which side does it take? (API judge)

The key comparison is between adapters: the us_liberal and us_conservative models should
diverge on the human-data tests in the directions the survey data predicts.
