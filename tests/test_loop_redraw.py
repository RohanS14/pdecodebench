"""Redraw path: a looped draw is replaced by a fresh sample, not a continuation."""
import inspect

from crossmodal.eval import backfill_no_verdict as BF


def test_redraw_is_a_fresh_sample_not_a_continuation():
    src = inspect.getsource(BF.redraw_looping)
    # the continuation path builds `prefix + response`; the redraw path must not
    assert "prefix + " not in src
    assert 'prompt = tok.apply_chat_template(' in src
    # response is REPLACED, not appended to
    assert '"response": text,' in src


def test_redraw_moves_only_the_seed():
    """Same prompt, same sampling params. A temperature or repetition-penalty change
    would make this a different treatment rather than another draw."""
    src = inspect.getsource(BF.redraw_looping)
    assert "gen=gen" in src
    for forbidden in ("temperature=", "repetition_penalty", "frequency_penalty",
                      "presence_penalty"):
        assert forbidden not in src, forbidden


def test_redraw_takes_the_first_terminating_sample():
    """n=1 and no scoring of candidates against each other: picking the 'better' of
    several samples, or retrying until a draw agrees with the other two, would be
    selecting on the outcome."""
    src = inspect.getsource(BF.redraw_looping)
    assert "n=1" in src
    assert "localization_correct" not in src
    assert "detection_correct" not in src


def test_last_attempt_is_written_even_if_it_still_loops():
    """A slot that keeps looping must stay in the arm as a no-verdict row rather
    than disappearing, or the merged arm silently loses draws."""
    src = inspect.getsource(BF.redraw_looping)
    assert "attempt < args.redraw_attempts" in src
    assert '"redraw_still_looping"' in src


def test_redraw_and_repair_loops_cannot_both_claim_the_loops():
    src = inspect.getsource(BF.main)
    assert "args.redraw_loops and not args.repair_loops" in src


def test_redraw_seed_cannot_collide_with_a_continuation_seed():
    assert BF.REDRAW_SEED_STRIDE > 100000
    src = inspect.getsource(BF.redraw_looping)
    assert "REDRAW_SEED_STRIDE * attempt" in src


def test_defaults_leave_loops_untouched():
    """The flag must be opt-in: every arm already produced was built without it."""
    import argparse
    import os
    os.environ.pop("REDRAW_LOOPS", None)
    p = argparse.ArgumentParser()
    # mirror the registration in main()
    p.add_argument("--redraw_loops", action="store_true",
                   default=os.environ.get("REDRAW_LOOPS", "") == "1")
    assert p.parse_args([]).redraw_loops is False


def test_retry_is_gated_on_no_verdict_not_on_the_loop_detector():
    """A redraw that ruminates must be resampled, not accepted.

    The retry used to read `if looping and attempt < args.redraw_attempts`, so a
    fresh sample that ran to the exact token cap while still producing novel text --
    rumination, no repeated 12-gram, is_looping() False -- was written on attempt 1
    and never retried. Measured on 2026-08-23: GLM sat at 62 residual draws, every
    one at exactly 65,536 output tokens with the detector reporting zero loops, so
    raising --redraw_attempts could not have moved it. The predicate that matches
    what the report drops is "ended without a verdict", not "looks like a loop".
    """
    src = inspect.getsource(BF.redraw_looping)
    assert "if no_verdict and attempt < args.redraw_attempts:" in src, \
        "retry must be gated on no-verdict"
    assert "if looping and attempt < args.redraw_attempts:" not in src, \
        "the loop-detector gate is the bug this test exists to prevent"
    # and the no-verdict test must be the same two-part one used everywhere else
    assert 'str(cand.finish_reason) == "length"' in src
    assert '"</think>" not in text' in src


def test_no_verdict_predicate_agrees_between_selection_and_retry():
    """The rows selected for repair and the rows retried must be the same rule.

    If needs_backfill() and the retry gate ever diverge, the pass would resample a
    set the report does not drop, or stop short of one it does.
    """
    sel = inspect.getsource(BF.needs_backfill)
    retry = inspect.getsource(BF.redraw_looping)
    for token in ('"length"', '"</think>"'):
        assert token in sel and token in retry
