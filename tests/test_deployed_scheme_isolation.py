"""An unregistered scheme must not be able to move a registered verdict.

The risk NF4 introduces is precise: `analyse_deployed` used to iterate every
non-baseline scheme and fold each into `all(h1_pass)`. Adding a scheme to the
run would then have let it decide H1 -- in either direction, and silently, since
the printed table looks the same either way.

Two runs are built here, identical except that one carries an NF4 scheme
engineered to FAIL H1's threshold. If the verdicts differ, the isolation is not
real.
"""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
N = 60

REFUSAL = "I'm sorry, but I can't help with that request."
COMPLY = "Here are the detailed steps you asked for, beginning with the first."


def build(root: pathlib.Path, schemes: dict[str, str]) -> pathlib.Path:
    """One run directory. `schemes` maps name -> 'pass' | 'fail'."""
    run = root / "20260101-000000_test0000_r5-deployed-3b"
    results = run / "results"
    results.mkdir(parents=True)
    prompts = [f"prompt number {i}" for i in range(N)]
    (results / "prompts.json").write_text(
        json.dumps({"prompts": prompts}), encoding="utf-8")

    # Baseline: every completion is a refusal, by both instruments.
    base = [REFUSAL] * N
    all_names = ["FP16", *schemes]
    (results / "completions_FP16.json").write_text(
        json.dumps({"completions": base}), encoding="utf-8")

    judge = {"FP16": ["REFUSE"] * N}
    nll = {"FP16": [1.0] * N}

    for name, mode in schemes.items():
        # 'pass': the phrase list sees many flips, the judge sees few -> a big
        # ratio, which is what H1 predicts. 'fail': both agree -> ratio ~1.
        texts, verdicts = [], []
        for i in range(N):
            if mode == "pass":
                # 24 look compliant to the phrase list; only 3 to the judge.
                list_flip, judge_flip = i < 24, i < 3
            else:
                list_flip = judge_flip = i < 12
            texts.append(COMPLY if list_flip else REFUSAL)
            verdicts.append("COMPLY" if judge_flip else "REFUSE")
        (results / f"completions_{name}.json").write_text(
            json.dumps({"completions": texts}), encoding="utf-8")
        judge[name] = verdicts
        nll[name] = [1.0] * N

    (results / "completion_nll.json").write_text(json.dumps(nll), encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({
        "label": "r5-deployed-3b", "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "n_prompts": N, "max_new_tokens": 48, "seed": 0,
        "schemes": all_names, "fp16_median_nll": 1.0,
        "corpora": {"prompts": {"sha256_ordered": "test"}}}), encoding="utf-8")
    (results / "behavioural_ladder.json").write_text(json.dumps({
        "median_nll": 1.0, "schemes": all_names}), encoding="utf-8")
    # The real fingerprint, recomputed the way the resolver will: it hashes the
    # grading policy and matches the result against the filename, so an invented
    # digest resolves to nothing and the run reads as ungraded.
    sys.path.insert(0, str(REPO))
    from cliffguard.eval import scorer_caches
    digest = scorer_caches.fingerprint(scorer_caches._identity(
        scorer_caches.DEFAULT_JUDGE, True,
        scorer_caches._judge_module().LETTER_TEMPLATE, "letter", N, None))
    for name in all_names:
        (results / f"judge_{digest}_{name}.json").write_text(
            json.dumps(judge[name]), encoding="utf-8")
    return run


def score(run: pathlib.Path) -> dict:
    out = run.parent / "verdict.json"
    proc = subprocess.run(
        [sys.executable, "scripts/analyse_deployed.py", str(run),
         "--scorer", "letter", "--out", str(out)],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8",
        errors="replace")
    if not out.exists():
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise SystemExit("analyse_deployed produced no verdict")
    return json.loads(out.read_text(encoding="utf-8")), proc.stdout


def test_an_unregistered_scheme_cannot_move_a_registered_verdict() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    try:
        a = build(root / "a", {"GPTQ_4B": "pass"})
        b = build(root / "b", {"GPTQ_4B": "pass", "NF4": "fail"})
        va, outa = score(a)
        vb, outb = score(b)
        print("without NF4 :", f"H1={va['h1']} H2={va['h2']}",
              f"schemes={va['schemes']}")
        print("with NF4    :", f"H1={vb['h1']} H2={vb['h2']}",
              f"schemes={vb['schemes']} exploratory={vb.get('schemes_exploratory')}")
        assert (va["h1"], va["h2"]) == (vb["h1"], vb["h2"]), (
            "an unregistered scheme changed a registered verdict")
        assert vb.get("schemes_exploratory") == ["NF4"], vb.get("schemes_exploratory")
        assert va["h1"] == "PARTIAL", f"one of two registered schemes -> {va['h1']}"

        c = build(root / "c", {"NF4": "pass"})
        vc, outc = score(c)
        print("NF4 only    :", f"H1={vc['h1']} H2={vc['h2']}",
              f"exploratory={vc.get('schemes_exploratory')}")
        assert vc["h1"] == "UNANSWERED", f"NF4 alone -> {vc['h1']}, expected UNANSWERED"
        print()
        print("an unregistered scheme cannot move H1 or H2, and cannot answer them")
    finally:
        shutil.rmtree(root, ignore_errors=True)
