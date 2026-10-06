# Findings log

Running notes on what we learn about the data and the models. Newest entries at the top of each
section. Add an entry whenever a check, filter or eval shows something non-obvious.

## Training data

### 2026-10-05: Planned transfer test: non-financial Equality/Proportionality -> financial MFQ-2 items

Problem: in v2, the strict overlap filter (cosine >= 0.52 to any human test item) removed most
of exactly the content added for Equality/Proportionality: income/wealth 33/40, pay-by-effort
28/38, equal-shares-regardless-of-effort 25/40. These facets are what the (financial) MFQ-2
Equality and Proportionality items ask about, so any on-topic dilemma lands near them.

Decision (user, 2026-10-05): sharpen the two constructs and train them **without money**.
Equality = same treatment, standing, say and share regardless of contribution, including
resisting domination; Proportionality = what you get out matches what you put in (effort, merit,
contribution; free-riders forfeit; punishment fits the offense). New non-financial facets:
voice, same rules for all, turns/access, credit, respect/status, anti-domination vs credit by
contribution, say by stake, earned privileges, playing time by merit, free riders, proportional
punishment (`mft.generate_v2.NONFIN_FACETS`; half the jobs pit Equality directly against
Proportionality). The strict 0.52 filter stays.

**This makes MFQ-2 Equality/Proportionality a cross-domain transfer test:** train on
non-financial equality (equal speaking time, same rules for the director and the janitor), test
on financial items ("everyone should earn the same", "raises should reflect effort"). Report it
as its own result in round 2: does the liberal-vs-conservative difference on these items move
compared with round 1, where the facets were absent?

### 2026-10-05: How much to trust the embedding overlap measure (text-embedding-3-small)

No ground truth for "too close", so three validity checks on cached vectors:
- MFQ-2 items: same-foundation pairs average cosine 0.41 vs cross-foundation 0.25, so the
  embeddings carry foundation content.
- Vignettes sorted into their foundation by nearest same-foundation vignettes (leave-one-out,
  6 classes, chance ~17%): **84% correct**. They track moral topic well.
- **Topic, not stance:** opposite-stance answers to the same dilemma average 0.77, answers to
  different dilemmas 0.34. The measure cannot tell "argues for equal shares" from "argues for
  proportional shares" in the same setting; it only sees "same topic".
- So the overlap filter is a reasonable guard against *thematic* leakage and blind to whether a
  training example argues a test item's position. The 0.52 cutoff was set by inspecting v1's
  top tail (p99 0.527, all reward-splitting near MFQ-2 items 9/15/26), not from a standard.

### 2026-10-05: v2 generation pilot: facet-balanced + believer's voice, with/without real seeds

60 jobs (10 per foundation), gpt-5.5, identical jobs in both arms; v1 baseline = 60 matched
dilemmas from the training set. Seeds: Social Chemistry 101 situations + MFRC comments matched to
the facet by embedding (`mft.seeds`). Scores: `data/v2/pilot_scores.json`.

| metric | v1 | v2 no seed | v2 seed |
|---|---|---|---|
| intended facet hit | - | 0.90 | 0.83 |
| distinct facets (of 35) | 19 | 33 | 32 |
| blind label OK, mini / DeepSeek | 0.98 / 0.92* | 0.98 / 0.97 | 0.93 / 0.97 |
| authenticity 1-5, mini / DeepSeek | 4.50 / 4.75 | 4.74 / 4.86 | 4.59 / 4.89 |
| mean pairwise cosine of prompts (lower = more varied) | 0.291 | 0.307 | 0.281 |
| median max-cosine to human test items | 0.35 | 0.38 | 0.39 |

\* v1 sample was drawn from already-verified data, so its label score is inflated.

- Facet balancing fixes coverage (19 to 32-33 facets) with high facet accuracy; zero refusals in
  120 charged-facet generations (patriotism, chastity, income equality, etc.).
- Seeds add variety (most varied arm) at a small facet-accuracy cost; facet forcing alone
  slightly homogenizes.
- **LLM authenticity ratings are saturated** (4.5-4.9 everywhere, 97-100% "argues from own
  value"), so they cannot detect subtle smoothing. Need pairwise comparisons and/or human ratings.
- v2 sits closer to the human test items, as expected when covering their facets; decontamination
  still applies (1/60 seeded prompts >= 0.52).
- Plan: v2 full set as a 50/50 mix of both arms; replace absolute authenticity with blind pairwise
  preference; collect a small human rating set.

### 2026-10-05: Facet coverage audit: generated data vs human test sets (`mft.coverage`)

Every generated dilemma side (n=2,004), MFQ-2 item (36) and vignette (99) was classified into
sub-themes of its foundation by gpt-5.4-mini. Full tables: `logs/coverage_report.md`; labels:
`data/analysis/coverage_units.jsonl`.

| foundation | training mostly | MFQ-2 measures | vignettes measure | biggest gaps (training share) |
|---|---|---|---|---|
| Care | protect vulnerable 40% | emotional suffering 83% | cruelty/mockery 56%, animals 25% | animals 0%, cruelty 6% |
| Equality | same rules / no favoritism 54% | equal income/wealth 67% | no favoritism 78% | income equality 0% (1/333) |
| Proportionality | earned deals kept 51% | pay by effort 50% | cheaters punished 75% | cheaters 0%, pay-by-effort 7% |
| Loyalty | team cohesion 51% | country/patriotism 50% | betrayal 62%, country 25% | patriotism 0%, betrayal 1% |
| Authority | rules/procedure 60% | parents/elders 33%, tradition 33% | disrespect/subversion 53% | subversion 0%, parents/elders 1% |
| Purity | sacred/religious 50% | sexuality/chastity 50% | sexuality 53%, bodily disgust 35% | sexuality 2%, crude speech 0%; tests have 0% sacred |

- **The generator avoids charged content:** patriotism, income redistribution, sexuality and
  animal cruelty are nearly absent. It defaults to institutional, workplace-safe settings.
- **Conflicts of goods vs witnessed violations:** training pits two positive values against
  each other; vignettes are all about witnessing a violation. Violation facets (betrayal,
  subversion, cheating) are 0-1% of training.
- **Generated test splits share the blind spots** (same generator), so only the human tests can
  reveal facet gaps. Embedding check agrees: least-covered human items (max cosine 0.20-0.27) are
  animal cruelty, anti-patriotism and sexual vignettes. Median nearest-training similarity per
  foundation 0.35-0.43.
- **Consequence for interpretation:** MFQ-2 movement after SFT is transfer *across facets* (e.g.
  bureaucratic-rules training shifting "obey parents" answers). Some per-foundation results are
  confounded by facet mismatch: liberal SFT *lowered* MFQ-2 Equality (3.07 to 2.67) although
  liberals score it 3.35, and MFQ-2 Equality is income equality, which training never covered.
- **Fix for v2 data:** facet-balanced generation (one sub-theme assigned per job, balanced against
  the theory's facets, not the test items), including charged facets; decontamination unchanged.
- Caveats: LLM classifier labels; only 6 MFQ-2 items per foundation.

### 2026-10-04: Blind label check removed 62 of 1,080 dilemmas (5.7%)

Generator: gpt-5.5. Checker: gpt-5.4-mini, shown both responses in random order, not told the
labels. Rejects with verdicts: `data/generated/dilemmas.verified.rejected.jsonl`.

Confusion (intended rows, judged columns), full run:

| intended \ judged | Care | Equality | Proportionality | Loyalty | Authority | Purity |
|---|---|---|---|---|---|---|
| Care | 358 | 1 | 0 | 1 | 0 | 0 |
| Equality | 0 | 356 | 3 | 0 | 1 | 0 |
| Proportionality | 0 | 1 | 359 | 0 | 0 | 0 |
| Loyalty | **15** | 3 | 0 | 340 | 2 | 0 |
| Authority | 0 | 0 | 1 | 1 | 359 | 0 |
| Purity | 7 | 0 | 0 | 0 | **28** | 325 |

Two systematic generator failure modes:

- **Purity argued as Authority.** In sacred-space dilemmas (consecrated hall, burial cloth near the
  Torah table, menstruation and a prayer room) the "Purity" response justifies itself by obedience:
  "breaks long-standing rules", "the disciplined life he has pledged to keep", "a responsibility you
  were trusted to uphold". Real-world religious purity and authority overlap, so this boundary is
  genuinely blurry; training on these would teach Purity = rule-following.
- **Loyalty argued as Care.** When the Loyalty side is "favor my close friend who is going through
  a hard time" (breakup, rough month), the response argues from her suffering, not from the bond
  ("show someone close to you that she is not alone"). Loyalty reasoning should rest on the
  relationship itself.

Consequence: **Care vs Loyalty and Purity vs Authority are the least cleanly generated conflicts**
and are slightly underrepresented after filtering. Check these pairs specifically in the
post-training behavior eval.

Checker reliability: re-running the check on the 62 rejects found 37, then 41, still mismatched.
So roughly a third of rejects are borderline cases the checker flips on between runs. Cost of that
noise is only lost data (a false reject), never a mislabeled kept example.

### 2026-10-04: Equality vs Proportionality confusion fixed by prompt

Smoke test (before the fix): 2 of 7 "Equality" responses actually argued desert or promise-keeping
("she met the deal first"), which is Proportionality. Added an explicit definition of the boundary
to the generator prompt. Full run after the fix: 3 Equality→Proportionality and 1 reverse out of
720 Equality/Proportionality responses.

### 2026-10-04: Template repetition in early generations

Smoke-test dilemmas all ended "If they do X..., if they do Y... What should they do?" and responses
ended with "Still, ...". Risk: the model learns the template, not the values. Added anti-template
instructions (vary length 40-200 words, structure, names, openings and closings). Not yet
measured quantitatively.

### 2026-10-04: Overlap with human test items: no copies, 16 thematic near-matches removed

Embedded all generated text (text-embedding-3-small) and compared it with the 36 MFQ-2 items and
132 vignettes. No near-duplicates. The closest matches (cosine 0.52-0.63) are reward-splitting
dilemmas, the core Equality vs Proportionality conflict, near MFQ-2 item 26 ("share the rewards
equally, even if some worked harder"), item 9, item 15 and vignette mfv-041 (student gets an A
without doing his part). Dropped everything at cosine >= 0.52 (16 records) so these test items
stay unseen in theme as well as text. Similarity distribution: p50 0.376, p90 0.448, p99 0.527.

### 2026-10-04: Conservative profile yields ~40% less preference data

With a tie margin of 0.3 (1-5 scale), us_liberal decides 13 of 15 foundation pairs and
us_conservative only 8, because conservatives weight the foundations more evenly (a known MFT
result). Final DPO train sizes: us_liberal 728, us_conservative 441. If the conservative adapter
is weaker, consider a smaller margin for it.

Only 3 pairs flip between the two profiles: Equality vs Loyalty, vs Authority, vs Purity
(liberal picks Equality, conservative the other). Japan vs Egypt flips none; they differ in
overall level, not ranking, so they are a poor contrast for this method.

## Honesty and multi-agent behavior

### 2026-10-06: Value training changes which lies models condone (honesty dilemmas, eval only)

160 generated advice dilemmas (40 per lie type), answered by base, label SFT seeds 0-2 and no-label
seed 0 per profile; DeepSeek judged deceive / honest / mixed (`mft.honesty`). Share recommending
deception: base 0-2% for every type. Group-protective lies: **conservative 18% (label) / 28%
(no label) vs liberal 4% / 2%**. Lies to a boss or institution: conservative 14% / 12% vs liberal
2%. Selfish lies 1-3% for all. White lies 2-8% (both trained profiles). No training example
concerned honesty: the Loyalty-weighted profile generalized to condoning lies that protect one's
group. The authority-lie dilemmas mostly protect coworkers, so they likely load on Loyalty too
(the predicted Authority effect, more condemnation, is not seen).

### 2026-10-06: Multi-agent competition: training halves concealment toward users; peer treatment not robust

816 episodes (8 shared-resource scenarios x 3 identity cues x 2 samples; base-base, and
lib/con pairings for label seeds 0-2 and no-label seed 0), `mft.multiagent` on H100s. Honesty
judged by DeepSeek against the true log.
- **Hid own hostile action from the user** (among agents that claimed/locked/reverted/escalated):
  base 59% [43-73], label liberal 32-33%, label conservative 27-30%, no-label 19-45%.
- Lies to the peer: 0-4% everywhere. Escalation: base 22%, label 12-17%.
- Profile-specific peer treatment is not robust: with the label, con-con pairs cooperate more than
  lib-lib (agreement 62% vs 52%, exclusion 27% vs 38%); without the label (one seed) it reverses
  (lib-lib 62% vs con-con 42%, exclusion 53% for con-con).
- Identity cue: the 24-episode pilot's in-group/out-group effect (88% vs 38% agreement) did not
  replicate at 96 base agent-episodes (44% in-group vs 62% out-group). Trained models show small
  cue effects (conservatives exclude out-group peers 41% vs in-group 33%).


### 2026-10-06: Agent-action eval: values transfer to actions only partly, and asymmetrically

258 usable tool-use scenarios (130 with robust labels), Qwen tool-calling format, randomized tool
order, choice = first tool called. Models: base; label SFT seeds 0-2 and no-label SFT seed 0, per
profile. Base control accuracy 81%, label-s0 models 80% (no loss of agent competence).

| comparison | different action on the same scenario |
|---|---|
| same profile, different seed (noise floor) | 3-6% |
| liberal vs conservative, label s0 / s1 / s2 | 8% / 7% / 7% |
| liberal vs conservative, no label s0 | 11% |
| base vs trained (either profile) | 19% |
| (for comparison: written advice, liberal vs conservative) | 51% |

Direction (robust labels, clear preference): **conservative 62% -> 75-86%** choosing the
profile-preferred action (no-label 86%, label 82/75/78%); **liberal 57% -> 54-59% (no shift)**.
Flip pairs (Equality vs Loyalty/Authority/Purity, n~100), share choosing Equality: base 48%;
label models 42-46% for both profiles (no separation); no-label liberal 50% vs conservative 38%.

- Training changes agent actions (19% vs base) but mostly in a profile-independent way; the
  profile-specific part is ~2x seed noise, far smaller than in written advice.
- Asymmetric: the conservative profile transfers into actions, the liberal one does not.
  Untested hypothesis: workplace-agent contexts already pull the base model toward procedural,
  authority-respecting actions, which liberal advice-training does not override.
- The label does not help actions (agents almost never emit it: 0-2%); no-label models diverge
  more (11% vs 7-8%).
- Label-before-tool-call: 0-1.7%, so the first-token habit does not break tool use.

### 2026-10-05: Foundation-specific action guidance fixes Purity, improves Loyalty

Added "how each foundation looks as an action" guidance to the scenario generator (Loyalty = the
bond with our own people, not obeying the org or helping the suffering; Purity = clean/undefiled,
disgust, sexual impropriety, crude, unnatural, not health or rules). Same 12 specs, gpt-5.5,
3 votes per checker. Share of judgments matching the intended action label (mini / DeepSeek):
Purity 46% (earlier, all authors) to **89% / 89%**; Loyalty 43% to **67% / 67%**; Authority
73% / 87%; Care 83% / 100%; Equality **50%** / 89%. New ambiguity: in Equality-vs-Purity
scenarios, "treat this person like everyone else" is read as Care (inclusion as kindness) by mini.
Pass by both checkers (majority votes): 5/12 (noisy at n=12).

### 2026-10-05: Wider bake-off (Kimi K3, GLM 5, Mistral Large 3); action labels are the bottleneck

Same 12 specs, all five authors re-checked together by both checkers (fresh run).

| author | pass both | mini | DeepSeek | note |
|---|---|---|---|---|
| gpt-5.5 | 4/12 | 8/12 | 5/12 | best |
| DeepSeek V3.2 | 2/11 | 3/11 | 3/11 | |
| Kimi K3 | 1/12 | 1/12 | 6/12 | priciest Bedrock option |
| GLM 5 | 0/12 | 1/12 | 1/12 | action labels mostly wrong |
| Mistral Large 3 | 0/12 | 0/12 | 0/12 | value words in 6/12 tool descriptions |

- **Action-level foundation attribution is the bottleneck for every author.** Checker agreement
  with intended labels, pooled: Care 51/60, Authority 41/50, Equality 39/58, **Loyalty 17/40**
  (misread as Authority 14x: favoring long-time partners looks like deferring to the org),
  **Purity 13/28** (misread as Care 9x: shielding users from graphic content looks like harm
  prevention). Mirrors the training-data rejects (Loyalty->Care, Purity->Authority). The flip
  pairs (Equality vs Loyalty/Authority/Purity) are exactly where labels are least reliable.
- **Checker verdicts are unstable across runs:** on identical gpt-5.5 scenarios DeepSeek went
  10/12 -> 5/12 and mini 6/12 -> 8/12. Single-pass checks are not a reliable filter; use repeated
  checks with majority vote.
- Implication for the experiment: the primary action metric should be *divergence between the
  liberal and conservative adapters on the same scenario* (label-free), with direction predicted
  only on scenarios whose labels are robust.

### 2026-10-05: Author bake-off, DeepSeek V3.2 (Bedrock) vs gpt-5.5

Same 12 scenario specs (6 flip-pair, 4 other, 2 safety) from each author, after tightening the
generator prompt. Each set blind-checked by two checkers from different labs, so self-preference
would show as asymmetry. Pass = labels match + both options legitimate + not settled by
instructions + control answer correct.

| author | checker | pass | labels | legit | unsettled | control |
|---|---|---|---|---|---|---|
| DeepSeek V3.2 | gpt-5.4-mini | 3/11 | 5/11 | 9/11 | 9/11 | 7/11 |
| DeepSeek V3.2 | DeepSeek V3.2 | 2/11 | 5/11 | 9/11 | 10/11 | 7/11 |
| gpt-5.5 | gpt-5.4-mini | 6/12 | 7/12 | 11/12 | 12/12 | 12/12 |
| gpt-5.5 | DeepSeek V3.2 | 10/12 | 11/12 | 11/12 | 11/12 | 11/12 |

- **gpt-5.5 writes clearly better scenarios**, and DeepSeek rates gpt-5.5's work above its own,
  so this is not checker self-preference. DeepSeek's weaknesses: actions that don't cleanly honor
  the intended foundation (both checkers: 5/11 correct) and broken controls (7/11). DeepSeek also
  failed schema validation on 1 of 12.
- Price is not the deciding factor: DeepSeek is ~16x cheaper per output token, but at a 2/11 pass
  rate (both checkers) and weaker survivors, gpt-5.5 is the better buy.
- **Checkers disagree a lot on labels for gpt-5.5** (mini 7/12, DeepSeek 11/12): foundation
  attribution for actions is genuinely harder than for written arguments. Requiring both checkers
  to agree is a strict but defensible filter.
- **LLM "moral language" check was useless:** gpt-5.4-mini flagged 15 of 23 scenarios; a word
  search found zero value words in any prompt, task or tool description. Replaced with a word list.
- Both authors avoided moral vocabulary in tools entirely after the prompt fix.

## Human data and profiles

### 2026-10-04: Profiles reproduce published values

Japan and Egypt profiles computed from the Study 2 item-level data match the preprint's Table 7
means exactly (tested in `tests/test_data_pipeline.py`).

US liberal/conservative profiles come from Study 3, which has foundation totals only, no item
responses. The left/right direction of `porient_1` (1 = left) is inferred: in the data, Equality
predicts it negatively and Authority positively, matching the paper's coefficients.

MFQ-2 item 27 has two official wordings (preprint/code vs the OSF English .docx); we use the
preprint/code wording. See `data/raw/SOURCES.md`.

## Models

### 2026-10-06 (night): Prompting baseline, 14B, agentic training, oversight, extra conflict scenarios

- **Prompting moves words, not actions.** Untrained 7B with a persona system prompt (profile foundations in
  order, with survey weights): MFQ-2 gaps overshoot the survey (Authority +2.58, Loyalty +1.81, Purity +1.79;
  r = 0.93-0.97), advice agreement 68% / 61% (below trained 71% / 74%), agent-action split +1.0 [+0.2, +1.9],
  conflict split +1.0 [+0.4, +1.7], oversight split ~0. Group-protective lies 8% (con persona) vs 0%.
- **14B (no-label, seed 0):** MFQ-2 similar to 7B; advice 86% / 70%; agent split +11.1 [+9.0, +13.4] (7B +4.9);
  conflict split +20.2 [+18.1, +22.4] (7B +7.3), across nearly every type incl. stop orders (+24). Untrained 14B
  picks the deceptive option 20.6% of conflicts (up to 35% vs insiders); trained 11.0% (lib) / 14.7% (con).
- **Agentmix (advice + held-out tool-use):** agent split +5.3, conflict +7.4 (no-label +4.9, +7.3); Care vs
  Authority +12.1 vs +9.4. Multi-agent: con-con 33% agreement vs lib-lib 48% (no-label pattern; now 3 of 4
  setups); private advantage hardball repeats (exclusion 56-90%, misleads 23-29%).
- **Conflict scenarios, 296 total:** split label +5.6, no-label +7.3, round 3 +2.7, rank 64 +9.2 (all CIs > 0).
  Person vs AI: all |diff| <= ~1 pt. Outsider deception con > lib in all 4 conditions but only 2-6 pts.
- **Oversight (56 scenarios: shutdown, goal change, review gate; human or AI overseer):** every model complies
  71-78%; covert continue-while-appearing-to-comply 5-9% for all incl. base; profile split only on review gates
  (+4 to +9); person vs AI overseer no difference.
- **Conversation judge v1 was unreliable** (flagged ~70% of base episodes as misleading, incl. honest closings,
  and counted "no messages before acting" as deception). Re-judging with a rubric limited to false statements
  and a concealed main outcome; v1 kept as `episodes.judged_v1_strict.jsonl`.
- **Neutral-fine-tuning control** (random winner per dilemma, same data/format, 2 seeds) queued to separate
  value effects from fine-tuning effects for trained-vs-base claims (hidden-info hardball, concealment, format).

### 2026-10-06 (evening): Round-3 multi-agent; agentic training (agentmix) first readouts

- **Round-3 multi-agent (after bare-JSON parse fix; 41% turns still unparsed, mostly prose narrating an action):**
  con-con agreement 43% / exclusionary 31% vs lib-lib 65% / 16%. Same direction as round-1 no-label
  (38/44 vs 58/28). Round-1 *label* models are the exception (con-con 69/27). The earlier "label flips peer
  treatment" reading is wrong; the pattern holds in 2 of 3 training setups. Private advantage (r3): con excludes
  more (54% vs 31%), lib misleads more (27% vs 17%): mixed.
- **Agentmix (no-label advice + held-out tool-use scenarios, 152-240 agent examples, seeds 0-1):** MFQ-2 gaps
  equal or larger than no-label advice-only (seed 0: Loyalty +0.27, Authority +0.24, Purity +0.57; seed 1 like
  advice-only); advice agreement unchanged (73% / 71-76%). Adding actions does not cost stated values.
  Agent-test readouts (agent_lp, conflict, multi-agent) running.
- Ops: new pods lacked `logs/`, so tmux jobs died at the redirect and pods idled ~25 min; setup now creates it.

### 2026-10-06: Conflict decision points (agent vs person / AI agent): profiles split as predicted; counterpart type barely matters; conservative agents deceive outsiders more

150 verified scenarios (`data/conflict/scenarios.verified.jsonl`; 10 conflict types; each in a person and
an AI-agent-for-that-person version), P(each tool) at the decision point for base + 18 adapters
(`results/conflict/scores.jsonl`, `report.txt`). Actions: two open actions (foundation A / B), a deceptive
action that goes against the other party, ask_user.

- **Profile divergence, predicted direction:** label +3.7 [+2.0,+5.5] pts, no-label +4.5 [+2.9,+6.2], rank-64
  +6.3 [+4.1,+8.7], round 3 +0.9 (ns; its tool use is degraded, see below). Driven by low-status (+10 to +24:
  conservative defers to the senior claimant), outsider (+6 to +14: favors the in-group), cheater (no-label +11).
  Senior request (ceiling: every model defers 76-94%), sanctity and merit ~0.
- **Person vs AI counterpart: no meaningful difference** (all |diff| <= 2.2 pts, every model incl. base), and the
  liberal-conservative gap is the same for both (~-1 pt). Single decisions do not treat AI agents as a different
  moral category.
- **Deception is targeted down the hierarchy** for every model (15-24% toward outsiders / low-status vs 2-5%
  toward seniors). Conservative > liberal deception toward outsiders in all four conditions (24.3 vs 16.8,
  23.7 vs 16.6, 22.9 vs 17.5, 23.3 vs 17.9%). Training cuts deception of overseers (base 8.6% -> 3-6%) but
  raises it toward insiders asking favors (2.6% -> 8-11%).
- Verification note: the LLM check for "versions differ only in the counterpart" was unreliable (rejected items
  whose notes said they differed only in the counterpart); replaced by text similarity >= 0.75 (median 0.94).
- **Round-3 models drop the <tool_call> tags** and write bare JSON (82% of multi-agent turns unparsed); after
  accepting bare JSON 35% remain, mostly prose that narrates an action without calling a tool. Round-3
  multi-agent runs are being redone; the log-prob scoring here forces the tag and is unaffected.

### 2026-10-06: Rank 64 vs 16 (no-label, seed 0): capacity is not the bottleneck

MFQ-2 gaps (con-lib) at rank 64 sit inside the rank-16 seed range (Loyalty +0.11 vs +0.05..+0.18, Authority
+0.17 vs +0.11..+0.13, Purity +0.28 vs +0.26..+0.44); eval loss 1.744 vs 1.768 (better text fit). Liberal advice
agreement 79% vs 71-72% (one seed). Rank 64 gives the largest agent divergence in the conflict test (+6.3 vs +4.5),
also one seed. Small questionnaire gaps come from the data and advice-to-questionnaire transfer, not adapter size.

### 2026-10-06: Multi-agent re-run with the parse fix: peer treatment is seed-robust but flips with the label; concealment gap shrinks; private advantage unlocks hardball

All episodes re-run after the curly-quote fix (`results/multiagent/v2/`, 1,200 standard + 336 private-facts
episodes; unparsed turns now 8-17%, was 22-31%). Reports: `logs/multiagent_report_v2.txt`, `_private_v2.txt`.

- **Peer treatment depends on the label, consistently across seeds.** No-label: con-con agreement 33/33/48%
  (seeds 0/1/2) vs lib-lib 54/60/60%, exclusionary moves 69/71/54% vs 56/50/35%; mixed pairs in between, in
  every seed. Label: con-con 75/67/67% vs lib-lib 52/54/62%. The earlier "reversal" is real, not noise.
- **Concealment from the user, corrected:** base 38% [26-53] (was 59%), label 16-26%, no-label lib 15-19%,
  no-label con 27-37%. Trained agents still conceal less than base, but the gap roughly halved and CIs nearly
  touch. The old 59% was inflated by the parse handicap (trained agents lost turns, base did not).
- **Private advantage (A secretly knows its need is minor, B = base):** false urgency is ~50% for every model
  (base 56%, trained 44-55%). But trained agents, either profile, mislead the peer 21-31% vs base 11% and use
  exclusionary moves 68-94% vs base 31%. Without private info trained and base are similar (27-44% vs 34%).
  Value training on advice text makes agents play harder when they hold hidden information, regardless of profile.

### 2026-10-06: Targeted agent scenarios replicate the action divergence (no-label > label)

104 new agent scenarios (`data/agent/v3/targeted.annotated.jsonl`, written for the Equality flip
pairs and Care vs Authority), scored with `mft.agent_logprob` for base + 12 adapters on pod 5.
Divergence = P_lib(A) - P_con(A) in the direction the profiles predict (threshold 0.10, all 3 seeds same sign).

| set | cond | mean divergence [95% CI] | stable (predicted) | Care vs Authority stable |
|---|---|---|---|---|
| targeted only | no-label | +0.062 [+0.033, +0.094] | 27 (22) | 10 (10) |
| targeted only | label | +0.017 [-0.006, +0.040] | 25 (16) | 9 (7) |
| existing + targeted | no-label | +0.049 [+0.033, +0.065] | 102 (75) | 20 (20) |
| existing + targeted | label | +0.029 [+0.015, +0.043] | 82 (55) | 17 (13) |

Fresh scenarios reproduce both earlier results: the no-label models act on the profile more than
label models (whose effect on new scenarios is not significant), and Care vs Authority is the
cleanest split (20/20 stable divergences predicted, no-label). Robust-label items: 49/56 predicted.
Outputs: `results/agent_lp/divergence_{tag,notag}.json`, `divergence_targeted_{tag,notag}.json`.

### 2026-10-06: Silent-failure audit (all rounds). Round 3 invalid; multi-agent confounded by a parse bug

Audited every training log (60 on S3) and every saved output (`results/`, S3 `results/`) for NaN
metrics, degenerate text, judge failures and parse failures.

- **Round 3 (all four r3tag adapters) is invalid.** Training went NaN on the RunPod H100s
  (r3tag_s0/s1 liberal and s1 conservative logged `grad_norm: nan`; s0 conservative trained
  normally but its final eval loss was NaN), and the saved adapters emit only `!!!!`. Every
  round-3 number (behavior, MFQ-2, vignettes, `answers_r3` honesty, multi-agent) was computed on
  garbage and is discarded. The multi-agent CUDA assert on pod 3 was this NaN, not a sampling bug.
  Reproduced: round-3 data goes NaN within 10 steps on the same pod where round-1 data trains
  cleanly; no single example gives a NaN forward loss and nothing is truncated (max 407 tokens).
  Root cause: SDPA attention on the RunPod stack (torch 2.11, transformers 5.18, H100). A step-by-step
  hunt caught a micro-batch of ordinary, right-padded text with a *finite* loss (0.62) but NaN
  gradients: the classic padded-row NaN in the attention backward. Round 3's wider length mix makes
  such batches more common. Same seed, same batches with `attn_implementation="eager"`: no NaN, and
  ~8x faster (SDPA here was CPU-bound with the GPU near 0%). Training now uses eager. Rounds 1-2
  trained on AWS A10G with SDPA without NaN; their outputs pass the audit. Round 3 retrained on
  pods 3 and 5; the invalid artifacts were moved to `s3://.../invalid_nan_r3/`.
  Training now stops on any non-finite loss/grad (`StopOnNaN`), checks the saved weights are finite,
  and `run_job.sh` runs `mft.sanity` (4 generations, fail if degenerate) before any eval.
- **Rounds 1-2 are clean.** No NaN in any log; no degenerate outputs. Judge failures: the 51
  OpenAI-outage failures in notag_s1 conservative were re-judged (DeepSeek); otherwise 0-9 of 160
  per model came back unparseable from the judge and are excluded, as before.
- **Greedy agent eval: truncated tool calls were scored "none".** Calls with long argument lists
  hit the 256-token cap, so the JSON never closed. 2-30 of 258 conflict items per model (most in
  label models). Parser now reads the tool name from a cut-off call; re-scored from saved outputs
  into `results/agent_rescored/`. Conclusions unchanged: lib-vs-con disagreement 8.1/6.4/7.4%
  (label s0-s2) and 11.6% (no-label s0), vs 8.3/6.8/7.4/11.3% before; agreement with own profile
  moves at most 1.7 points. The log-prob "leaning" metric never parsed text and is unaffected.
- **Multi-agent: fine-tuned agents lost 18-34% of their turns to a parse failure; base lost 0%.**
  Fine-tuned models close JSON strings with a curly quote (`...today?”}}`), learned from the
  typographic quotes in the generated training answers. The turn was logged and the agent told
  to retry, but it lost the move. This confounds every base-vs-trained multi-agent comparison
  (including "hid own hostile action: base 59% vs trained 27-33%") and may skew lib-vs-con.
  Parser now straightens a curly quote that ends a string value (unparsed turns 18-34% -> 9-22%;
  the rest is mostly prose with no tool call, which is real behavior) and the per-turn cap is
  320 tokens (was 200). Multi-agent runs pod1/pod2/pod4_nolabel need rerunning; the private-facts
  run on pod 4 was restarted on the fixed parser.
- Also seen: label-trained agents invent tool names that start with their label
  (`proportionally_split`, `proportionate_revert`): the label reflex leaking into tool calls.

### 2026-10-06: Round 2 (facet-balanced v2 + non-financial Eq/Prop): Equality transfer works, Purity divergence reverses

Label SFT on `data/v2/dilemmas_round2.jsonl` (1,037 v2 + 197 non-financial Eq/Prop + 242 controls),
2 seeds per profile, evaluated on the round-1 held-out sets (re-judged by DeepSeek, 98.2% agreement
with in-run verdicts).

- **Cross-domain transfer test (planned 2026-10-05) succeeded:** MFQ-2 Equality (financial items),
  conservative minus liberal: round 1 +0.24 (wrong direction) -> **round 2 -0.42** (seeds -0.46,
  -0.39; survey -0.22). Liberal Equality 2.48 -> 2.80 (toward 3.35), conservative 2.72 -> 2.38.
  Training only on non-financial equality (voice, same rules, credit, turns, anti-domination)
  moved answers to "everyone should earn the same"-type items.
- **Binding divergence weakened/reversed:** Loyalty +0.17 -> -0.08, Authority +0.18 -> +0.02,
  **Purity +0.33 -> -0.19** (both seeds). Not explained by the training signal balance (win/loss
  counts per foundation have similar ratios in both rounds; liberal Purity loses 0/267 in r1 and
  0/370 in r2). The change is almost entirely the **liberal** model rating sexual-purity items
  higher in r2: virginity 1.64 -> 2.55, chastity 2.04 -> 2.87, natural medicine 2.46 -> 2.95,
  body-as-temple 2.80 -> 3.21; conservative model unchanged (virginity 1.96 -> 1.97).
- **Hypothesis (untested): tone of the winning answer.** v2 dilemmas often present purity-holders
  sympathetically (first-person, believer's voice), and the liberal-side advice validates them
  while choosing Care/Loyalty ("your modesty is not an insult"). Training may teach "override
  purity but respect it" in r2 vs "dismiss it" in r1. If true: what post-training teaches about a
  value depends on how the losing side is treated, not only on which side wins.
- Behavior on round-1 held-out dilemmas fell 85% -> 78% (both profiles): expected distribution
  shift (r2 data differs in style and generator setup from the r1 test set).

### 2026-10-06: Overnight ablation results (16 runs, all behavior re-judged by DeepSeek V3.2)

All 3,040 behavior answers re-judged with one judge (`mft.rejudge`); agreement with the original
verdicts 93.7% (n=2,894), consistent with the ~5% judge noise measured from base-vs-base. Metric:
free-generation answers siding with the profile on **clear-preference pairs** (|dw| >= 0.3),
held-out tools domain; mean over seeds [95% bootstrap CI over pooled items] (per-seed %).

| | us_liberal | us_conservative |
|---|---|---|
| untrained base | 52% [41, 64] | 65% [50, 79] |
| no label, SFT | 71% [65, 78] (70, 71, 73) | 85% [78, 90] (85, 83, 85) |
| **label, SFT** | **85% [80, 90]** (89, 86, 80) | **86% [79, 92]** (90, 82, 85) |
| label, SFT + DPO (1e-5, 1 ep) | 79% [73, 84] | 88% [81, 93] |
| label SFT + DPO w/ SFT term on unlabeled pairs (s0) | 87% [79, 94] (vs 90% SFT) | 83% [71, 93] (vs 87%) |
| label SFT + on-policy DPO w/ SFT term (s0) | 84% [76, 93] (vs 90% SFT) | 87% [76, 97] (vs 85%) |

- **Seed variance is small** (spread ~1-9 pts, typically ~3), so condition effects are real.
- **The first-token label helps for liberal (+14 pts, CIs don't overlap), not for conservative
  (+1).** Plausible reading: for conservative the no-label model already reaches ~85% on clear
  pairs; the label's extra commitment matters where training must override the base model's
  free-generation lean toward liberal positions. Untested.
- **No DPO variant beats SFT on behavior.** Plain DPO hurts (liberal -6); the SFT term reduces
  the harm; on-policy DPO had little to learn from: the SFT models already side with their
  profile in 89% (lib) / 84% (cons) of their own samples, 548/728 and 273/441 prompts were fully
  consistent, leaving ~190 pairs per profile. SFT is the working method; DPO is a documented
  negative result here.
- **MFQ-2 divergence is robust across seeds:** conservative-trained minus liberal-trained, no
  label (3 seeds): Loyalty +0.10, Authority +0.13, **Purity +0.32**, all seeds positive; label
  (3 seeds): Loyalty +0.17, Authority +0.18, Purity +0.33, but Care and Proportionality flip sign
  across labeled seeds. The label adds noise to questionnaire answering (consistent with its
  first-token habit interfering with the 1-5 rating format), while leaving the binding-foundation
  divergence intact.

### 2026-10-05 (late): Conservative "underperformance" is near-tie items, not weaker training

Behavior on held-out dilemmas, labeled runs pooled (pilot s0 + seeds 1-2 available so far),
split by how clearly the profile prefers one side (|w_a - w_b| >= 0.3 = clear):

| profile | clear pairs: base -> SFT | near-tie pairs: base -> SFT |
|---|---|---|
| us_liberal | 58% -> **87%** (n=213) | 100% -> 54% (n=26) |
| us_conservative | 61% -> **89%** (n=82) | 45% -> 67% (n=78) |

Where the profile has a clear preference, both models move about +28 pts to the same level. The
lower overall conservative number comes from its flat profile (Authority 3.91, Proportionality 3.82,
Care 3.80, Loyalty 3.73): about half of its test items are near-ties, where the survey barely
prefers either side and training skipped such pairs (min margin 0.3). **Headline behavior metric
should use clear-preference pairs; report near-ties separately.** Also noted: seed-to-seed spread
is ~1 pt (liberal no-label 71/71/72%, label 85/84%), so seed noise is small relative to the
condition effects.

### 2026-10-05: Why DPO hurts behavior here: likelihood displacement (and the fixes queued)

Conservative pilot DPO (5e-5) log: chosen log-prob -241 -> -284, rejected -286 -> -457, margin
12.8, loss 0.000. DPO widened the gap by pushing *both* answers down; the freed probability went
elsewhere, and free-generation behavior fell 80% -> 68%. This is "likelihood displacement"
(Razin et al. 2024, arXiv 2410.08847), driven by similar preferred/rejected pairs: our
opposite-stance answers to the same dilemma have embedding cosine 0.77. Also, in the tagged
condition DPO can win the objective through the first token: first-token P(profile side) rose
0.73 -> 0.86 while forced-tag steering fell 0.091 -> 0.073 (label and content decoupled). In the
untagged condition DPO was roughly neutral. Pairs are off-policy (gpt-5.5 text), which Tajwar et
al. 2024 (arXiv 2404.14367) argue makes the negative gradient less useful.

Queued fixes (results in `results/s3/tagrpo_s0`, `results/s3/onpolicy_s0`):
- `tagrpo`: labeled SFT, then DPO on unlabeled pairs with an SFT term on the chosen answer
  (TRL loss `sigmoid,sft`, the RPO/NLL recipe of Pang et al. 2024, arXiv 2404.19733).
- `onpolicy`: labeled SFT model samples 4 answers per training dilemma; DeepSeek V3.2 judges
  which value each sides with; DPO (+SFT term) on the model's own answer pairs. Prompts where all
  samples already side with the profile give no pair.
A user review of 30 off-policy pairs (does the "preferred" answer match the group?) is pending
(https://claude.ai/artifact/RJ8ex9M59u5BqMkGj9hT2g) to rule out bad pairs as a cause; note that
SFT trains on the same preferred answers and works, which argues against that.

### 2026-10-05: us_conservative full run; liberal vs conservative divergence; DPO overfits at 5e-5

**us_conservative (paired, 95% CIs; `python -m mft.stats us_conservative`):**
behavior siding with the profile 53% -> **80% after SFT** (+27 pts [+15, +38], McNemar
p = 0.0001; 25 items flipped toward, 4 away). This is the stronger test: the base model leans
liberal, so the conservative model had to move *away* from its default. First-token P(profile
side) 0.37 -> 0.73 (SFT) -> 0.86 (DPO). Vignette residual-vs-profile r ~ 0 throughout (no effect,
unlike liberal). MFQ-2 MAE got *worse* (0.28 -> 0.44): training lowers all MFQ-2 scores (as for
liberal) and the base model already sat at conservative levels.

**The two trained models diverge as predicted (SFT, paired by item):**

| MFQ-2 | conservative-trained minus liberal-trained [95% CI] | real survey difference |
|---|---|---|
| Authority | +0.29 [+0.22, +0.36] | +1.22 |
| Purity | +0.22 [+0.14, +0.30] | +1.52 |
| Proportionality | +0.18 [+0.09, +0.28] | +0.29 |
| Loyalty | +0.16 [+0.08, +0.24] | +1.16 |
| Equality | +0.09 [-0.02, +0.20] | -0.22 |
| Care | +0.02 [-0.04, +0.10] | -0.51 |

The four foundations conservatives score higher on all move in the right direction (CIs exclude
zero), at roughly 1/5 of the human gap. Care and Equality do not move in the predicted direction:
consistent with the coverage audit (MFQ-2 Equality = income equality, MFQ-2 Care = emotional
suffering; both near-absent in training). **The profile difference is the right readout:** both
trainings depress absolute MFQ-2 scores, so MAE-to-target is confounded by that shared shift.

Behavior: the two base runs (same model) agree on 95% of items, a direct estimate of judge noise.
After SFT the liberal and conservative models agree on only 49%. On the flip pairs (Equality vs
Loyalty/Authority/Purity, n=16) the liberal model sides with Equality 94% of the time, the
conservative model 12%.

**DPO learning rate:** 5e-6 did nothing (liberal); 5e-5 (conservative) overfit: loss 0.000 and
reward margin 12.8 by step 30, first-token P(side) up to 0.86 but free-generation behavior
*down* from 80% to 68%. DPO here optimizes the tagged likelihood margin at the expense of
behavior. Ablation jobs use 1e-5 for 1 epoch and also evaluate the SFT-only model, so their
conclusions do not depend on DPO tuning. Earlier seed-0 runs are kept as `pilot_s0` in S3; all
12 ablation runs (label/no-label x seeds 0-2 x 2 profiles) use one identical pipeline.

### 2026-10-05: Qwen2.5-7B us_liberal FULL run (2 epochs SFT, then DPO)

| eval | base | SFT | SFT+DPO |
|---|---|---|---|
| behavior: free generation sides with profile (n~80, held-out tools domain) | 0.63 | **0.82** | 0.79 |
| first-token P(profile side) within pair (test / held-out) | 0.49 / 0.44 | 0.87 / 0.90 | 0.88 / 0.91 |
| forced-tag steering per token | 0.02 | **0.16** | 0.16 |
| MFQ-2 MAE to liberal profile | 0.81 | **0.41** | 0.41 |
| vignettes residual vs profile, r (n=82) | -0.03 | **0.29** | 0.29 |
| vignettes r with human wrongness | 0.84 | 0.80 | 0.80 |

- SFT moves behavior, self-report and severity-controlled vignette judgments together. First
  evidence that conflict-choice training reaches free-form behavior on unseen dilemmas (63% to
  82% siding with the liberal ranking).
- **Statistics (`python -m mft.stats us_liberal`, paired, 95% bootstrap CIs over items):**
  behavior +19 pts [+6, +32], exact McNemar p = 0.011 (23 items flipped toward the profile, 8
  away); first-token P(profile side) +0.41 [+0.34, +0.48]; steering +0.14 [+0.12, +0.16];
  MFQ-2 MAE 0.81 [0.64, 0.98] to 0.41 [0.32, 0.52]; vignette residual-vs-profile r -0.03
  [-0.26, +0.18] to +0.29 [+0.06, +0.50]. All exclude zero for this run. Not yet captured: seed
  variance (one run per profile) and judge noise.
- **DPO did nothing, because it barely trained:** loss 0.694 to 0.641, reward margin 0.11 by step
  40. Learning rate 5e-6 is a full-fine-tune value; LoRA needs ~10x (5e-5). Fixed default; the
  corrected train.py reached the GPU before us_conservative's DPO stage started, so
  **us_conservative's DPO ran at 5e-5** and only us_liberal needs a DPO rerun
  (`scripts/rerun_dpo.sh`, results in `results/us_liberal_dpo2.jsonl`).

### 2026-10-05: Qwen2.5-7B us_liberal quick run, after 1 epoch of SFT (DPO crashed)

| eval | base | after SFT |
|---|---|---|
| first-token label mass (test / held-out) | 0.003 / 0.000 | 0.97 / 0.94 |
| first-token accuracy (test / held-out) | 0.10 / 0.00 | 0.95 / 0.95 |
| forced-tag steering per token | 0.0065 | 0.055 |
| MFQ-2 MAE to liberal profile (Pearson, n=6 so CI ~ -0.1 to 0.97) | 0.81 (0.72) | 0.45 (0.78) |
| vignettes: Pearson with human wrongness | 0.84 | 0.78 |
| vignettes: Spearman with profile (n=4, uninformative) | -0.2 | 0.4 |

- One epoch of SFT moved the MFQ-2 self-report substantially toward liberal survey means
  (Equality, Loyalty, Authority, Purity all down; MAE 0.81 to 0.45), although training never
  showed a questionnaire item. This is the first sign that conflict-choice training transfers
  to self-reported values.
- The tag now carries ~8x more steering, but still small in absolute terms.
- Agreement with *average* human wrongness ratings dropped slightly (0.84 to 0.78): judging
  like one group means judging less like the population average. Expected trade-off.
- **DPO OOM:** training completed (46 steps, ~14 s/step at batch 1 on A10G 24GB) but the
  end-of-epoch eval used the default eval batch size of 8 and ran out of memory before saving.
  Fixed: eval batch size now follows `--batch-size`.
- Statistical note: n=6/n=4 correlations are near-uninformative; full runs now save per-item
  results (`*.items.jsonl`) for bootstrap CIs and paired tests, and add a per-vignette
  severity-controlled metric (`residual_vs_profile_r`, n=82).

### 2026-10-05: Untrained Qwen2.5-7B-Instruct baseline (A10G, quick mode)

- **MFQ-2 self-report:** Care 4.80, Proportionality 4.10, Authority 3.92, Loyalty 3.72, Purity 3.11,
  Equality 3.07. Shape correlates with every profile (r = 0.72 liberal, 0.77 conservative, 0.82
  Japan, 0.65 Egypt), but its **absolute level is closest to US conservatives** (MAE 0.28 vs 0.81
  for liberals). It rates everything high and ranks **Equality last**, which is neither US group's
  pattern. The liberal "lean" seen in free-generation behavior does not show up in its
  questionnaire answers.
- **Vignettes:** wrongness ratings correlate **r = 0.84 with human ratings**; it rates the
  non-moral Social Norms vignettes near zero (0.43 vs ~2.6-3.3 for moral ones). Off-the-shelf, it
  already judges these stimuli much like people do.
- **First token:** puts ~0% probability on starting with a label; the tag carries no steering
  (0.0065 nats/token). Expected before training.
- **Behavior (n=10):** 8/10 side with the liberal profile.

**Eval-sampling bug found:** eval files were in generation order (grouped by foundation pair), so
`--limit 10/20` in quick mode tested only Care-vs-Equality/Proportionality dilemmas. Fixed in
`build_datasets.py` (deterministic shuffle) after this run; this run's quick-mode behavior and
first-token numbers are Care-pair-only. Full runs use limit 100 of 105 and were barely affected.

### 2026-10-04: Mac rehearsal, Qwen2.5-0.5B, us_liberal, SFT only (1 epoch). Pipeline check, not a result

Tiny model, fp32 on Apple MPS, QUICK mode (first-token on 20 items, behavior on 10). Treat every
number as a smoke signal only. DPO stage was stopped (swapping); outputs in `results_smoke/`.

| eval | base | after SFT |
|---|---|---|
| first-token label mass (test / held-out domain) | 0.003 / 0.001 | 0.92 / 0.86 |
| first-token accuracy (test / held-out domain) | 0.05 / 0.00 | 0.90 / 0.95 |
| P(profile's side) within the pair (test / held-out) | 0.50 / 0.55 | 0.81 / 0.78 |
| forced-tag steering per token | 0.011 | 0.022 |
| MFQ-2 Pearson with us_liberal profile (MAE) | 0.33 (0.67) | 0.58 (0.63) |
| vignettes Spearman with profile (4 foundations) | -1.0 | 0.8 |
| behavior agreement with profile (n=10) | 0.9 | 0.8 |

Observations worth checking at full scale:

- The reflex is learned fast and **transfers to the held-out "AI assistant with tools" domain**
  (0.95 accuracy) even at 0.5B after one epoch.
- **The base model already sides with the liberal profile 9/10 times** in free generation. If that
  holds at 7B, us_liberal is a weak test of whether training changes behavior (little headroom);
  **us_conservative is the more informative profile**, because it asks the model to move away
  from its default.
- The tag barely steers the continuation yet (0.011 to 0.022 nats/token), consistent with the
  "tag becomes decorative" risk. Watch this after DPO and at 7B.
- MFQ-2 scores all *dropped* after SFT (overall level down ~0.4) while the shape moved toward the
  profile (r 0.33 to 0.58). Level vs shape may move independently; report both.
