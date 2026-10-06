from mft.build_datasets import build, split_of
from mft.foundations import FOUNDATIONS, LABELS, foundation_pairs
from mft.generate_dilemmas import DOMAINS, DilemmaBatch, dilemma_jobs
from mft.llm import strict_schema as _strict_schema
from mft.profiles import load_profiles, load_weights, preference_prob, profile_from_responses, winner

W = {"Care": 4.6, "Equality": 4.2, "Proportionality": 3.4, "Loyalty": 2.8, "Authority": 2.4, "Purity": 4.5}


def test_labels():
    assert len(FOUNDATIONS) == 6 and LABELS[-1] == "None"
    assert len(foundation_pairs()) == 15


def test_preference_and_winner():
    assert preference_prob(W, "Care", "Authority") > 0.9
    assert abs(preference_prob(W, "Care", "Authority") + preference_prob(W, "Authority", "Care") - 1) < 1e-9
    assert winner(W, "Care", "Authority") == "Care"
    assert winner(W, "Care", "Purity") is None  # within margin


def test_profiles_config_loads():
    profiles = load_profiles()
    assert "us_liberal" in profiles and profiles["us_liberal"]["weights"]
    assert set(load_weights("us_liberal")) == set(FOUNDATIONS)


def test_profile_from_responses():
    item_map = {"q1": "Care", "q2": "Care", **{f"x_{f}": f for f in FOUNDATIONS if f != "Care"}}
    rows = [
        {"ideo": "liberal", "q1": "5", "q2": "3", **{f"x_{f}": "2" for f in FOUNDATIONS}},
        {"ideo": "liberal", "q1": "3", "q2": "3", **{f"x_{f}": "4" for f in FOUNDATIONS}},
        {"ideo": "conservative", "q1": "1", "q2": "1", **{f"x_{f}": "5" for f in FOUNDATIONS}},
    ]
    prof = profile_from_responses(rows, item_map, {"ideo": "liberal"})
    assert prof["Care"] == 3.5 and prof["Loyalty"] == 3.0


def _records():
    return [
        {"id": "d1", "kind": "dilemma", "domain": "family", "prompt": "P1",
         "responses": {"Loyalty": "stick with brother", "Care": "protect the child"}},
        {"id": "d2", "kind": "dilemma", "domain": "family", "prompt": "P2",
         "responses": {"Care": "c", "Purity": "p"}},  # tie under W -> skipped for training
        {"id": "c1", "kind": "control", "domain": "school", "prompt": "Q", "answer": "A"},
    ]


def test_build():
    sft, dpo, evals = build(_records(), W, min_margin=0.3)
    assert [r["id"] for r in sft] == ["d1", "c1"]
    assert sft[0]["completion"][0]["content"] == "Care\n\nprotect the child"
    assert sft[1]["completion"][0]["content"].startswith("None\n\n")
    assert dpo[0]["chosen"][0]["content"].startswith("Care") and dpo[0]["rejected"][0]["content"].startswith("Loyalty")
    assert len(evals) == 3  # ties still evaluated, with a soft target
    assert abs(sum(evals[1]["target"].values()) - 1) < 1e-9


def test_split_holdout_domain():
    r = {"id": "x", "domain": "agentic"}
    assert split_of(r, 0.1, "agentic") == "holdout_domain"
    assert split_of(r, 0.1, None) in {"train", "test"}


def test_jobs_and_schema():
    jobs = dilemma_jobs(per_cell=2, seed=0)
    assert len(jobs) == 15 * 2 * len(DOMAINS)
    schema = _strict_schema(DilemmaBatch)
    assert schema["additionalProperties"] is False
    assert all(d["additionalProperties"] is False for d in schema["$defs"].values())


def test_profiles_from_real_data_match_published_table():
    import csv
    from mft.profiles import _matches

    rows = list(csv.DictReader(open("data/raw/mfq2_study2_raw.csv", encoding="utf-8-sig")))
    items = {r["column"]: r["foundation"] for r in csv.DictReader(open("data/raw/mfq2_items.csv"))}
    japan = profile_from_responses(rows, items, {"country": "Japan"})
    # Atari et al. preprint Table 7: Japan C 3.03, Eq 2.27, Pr 3.14, L 2.66, A 2.67, Pu 2.63
    published = [3.03, 2.27, 3.14, 2.66, 2.67, 2.63]
    assert all(abs(japan[f] - p) <= 0.006 for f, p in zip(FOUNDATIONS, published))
    assert _matches({"x": "2"}, "x", "1-3") and not _matches({"x": "8"}, "x", "1-3")


def test_vignettes_parse():
    from pathlib import Path
    from mft.vignettes import SRC, parse

    rows = parse(Path(SRC).read_text(encoding="utf-8"))
    assert len(rows) == 132
    assert rows[0]["mfq2"] == "Care" and rows[0]["wrong"] == 3.4
