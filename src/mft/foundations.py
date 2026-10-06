"""The six MFQ-2 moral foundations (Atari et al., 2023) plus a non-moral control label.

The label word for each foundation is also the *first token* the model is trained to emit,
so keep them short, capitalised, single words that most tokenizers encode as one token.
`check_single_tokens` verifies this for a given tokenizer before training.
"""
from __future__ import annotations

FOUNDATIONS: dict[str, str] = {
    "Care": "Avoiding emotional and physical harm to others; kindness, compassion, protecting the vulnerable.",
    # Sharpened 2026-10-05: Equality is the same treatment, standing, say and share regardless of
    # contribution, including resisting domination; Proportionality is the link between what you
    # put in and what you get out. (Earlier: "Equal treatment and equal outcomes for individuals;
    # opposing unequal distribution or treatment." / "People being rewarded or punished in
    # proportion to their merit, effort or contribution.")
    "Equality": "Everyone gets the same treatment, standing, say and share regardless of what they contributed; resisting anyone ruling over, outranking or hoarding from the rest.",
    "Proportionality": "What you get out should match what you put in: effort, merit and contribution earn outcomes, and free-riding or cheating forfeits them; punishment fits the offense.",
    "Loyalty": "Standing with your group, family, team or nation; opposing betrayal.",
    "Authority": "Respecting legitimate authority, hierarchy and tradition; opposing subversion and disrespect.",
    # Rooted in disgust and contamination avoidance; the "sacred" sense extends that intuition. Clearest
    # in harmless-but-disgusting cases, which is what separates it from Care. (Revised 2026-10-05; the
    # first training set was generated with: "Sanctity and avoiding degradation or disgust; treating
    # the body, the sacred and the natural order as not to be defiled.")
    "Purity": "Disgust and contamination: keeping the body, food, sexuality and sacred things clean and undefiled; avoiding what is degrading, unnatural or unclean, even when no one is harmed.",
}

NONE_LABEL = "None"
LABELS: list[str] = [*FOUNDATIONS, NONE_LABEL]


def foundation_pairs() -> list[tuple[str, str]]:
    names = list(FOUNDATIONS)
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]


def label_token_ids(tokenizer) -> dict[str, int]:
    """First token id of each label as it appears at the start of an assistant turn."""
    ids = {}
    for label in LABELS:
        toks = tokenizer.encode(label, add_special_tokens=False)
        ids[label] = toks[0]
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"Label first-tokens collide for this tokenizer: {ids}")
    return ids


def check_single_tokens(tokenizer) -> list[str]:
    """Return labels that encode to more than one token (the first token still identifies them)."""
    return [l for l in LABELS if len(tokenizer.encode(l, add_special_tokens=False)) > 1]
