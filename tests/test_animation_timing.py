"""
Standalone CPU tests for nodes/improved_animation_processor.py
(Enhanced Animation Timing Processor).  No comfy imports; run with:

    /home/trent/ComfyUI/venv/bin/python tests/test_animation_timing.py
or
    /home/trent/ComfyUI/venv/bin/python -m pytest tests/test_animation_timing.py

Frames are synthetic: each "scene" is a fixed random-noise image, and a
hold repeats that scene N times.  Scenes are dissimilar to each other, so
the hybrid similarity metric splits them cleanly.
"""

import contextlib
import io
import os
import random
import sys

import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "nodes")
)

from improved_animation_processor import (  # noqa: E402
    AnimationDuplicateFrameProcessor,
    AnimationFrameRemover,
)

NODE = AnimationDuplicateFrameProcessor()
REMOVER = AnimationFrameRemover()


def scene(seed, size=32):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(size, size, 3, generator=g)


def build(pattern):
    """pattern: list of (scene_seed, hold_length)."""
    frames = []
    for seed, n in pattern:
        s = scene(seed)
        frames += [s.clone() for _ in range(n)]
    return torch.stack(frames)


def run(images, **overrides):
    kw = dict(
        similarity_method="hybrid", similarity_threshold=0.85,
        motion_tolerance=0.05, gray_style="desaturated", gray_intensity=0.5,
        preserve_first=True, preserve_last=True,
        preserve_global_first=True, preserve_global_last=True,
        skip_second=False, skip_second_to_last=False,
        min_sequence_length=2, min_gray_frames=4, insert_padding=True,
        align_keyframes=True, alignment_multiple=4, debug_info=False,
    )
    kw.update(overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        out, mask, report, removal = NODE.process_animation_timing(images, **kw)
    return out, mask, report, removal


def keyframes_from(report):
    line = [l for l in report.splitlines()
            if l.startswith("Final keyframe positions:")][0]
    return eval(line.split(":", 1)[1].strip())


def mask_rows(mask):
    return [int(mask[i].mean().round()) for i in range(mask.shape[0])]


def unique_scenes(pattern, min_len=2):
    """Scenes whose run is too short to count as a duplicate sequence."""
    return [scene(seed) for seed, n in pattern if n < min_len]


def is_untouched_unique(frame, pattern, min_len=2):
    return any(torch.allclose(frame, s) for s in unique_scenes(pattern, min_len))


def check_consistency(pattern, out, mask, report, min_len=2):
    """Invariants that must hold for every run:

    * every mask-black frame is either a listed keyframe or an untouched
      frame from a run too short to count as a hold
    * every listed keyframe is mask-black and pixel-identical to a scene
    * every mask-white frame has been altered (no leaked hold frames)
    """
    kfs = keyframes_from(report)
    rows = mask_rows(mask)
    all_scenes = [scene(seed) for seed, _ in pattern]
    for i in range(out.shape[0]):
        matches_scene = any(torch.allclose(out[i], s) for s in all_scenes)
        if rows[i] == 0:
            assert matches_scene, f"frame {i} is mask-black but altered"
            assert i in kfs or is_untouched_unique(out[i], pattern, min_len), \
                f"frame {i} is mask-black but neither keyframe nor unique"
        else:
            assert not matches_scene, f"frame {i} is mask-white but untouched"
    for k in kfs:
        assert rows[k] == 0, f"keyframe {k} is mask-white"


# ---------------------------------------------------------------- tests --

def test_global_first_and_last_outside_holds():
    """Frame 0 and the last frame are unique (not part of any hold).
    The global flags must still register them as keyframes."""
    pattern = [(1, 1), (2, 4), (3, 4), (4, 1)]
    imgs = build(pattern)
    out, mask, report, _ = run(imgs, insert_padding=False)
    kfs = keyframes_from(report)
    assert kfs[0] == 0
    assert kfs[-1] == out.shape[0] - 1
    assert torch.allclose(out[0], imgs[0])
    assert torch.allclose(out[-1], imgs[-1])
    assert mask_rows(mask)[0] == 0 and mask_rows(mask)[-1] == 0
    assert all(k % 4 == 0 for k in kfs), kfs
    check_consistency(pattern, out, mask, report)


def test_global_first_only_on_hold_start():
    """preserve_first off, global first on: frame 0 inside a hold stays."""
    pattern = [(1, 4), (2, 4), (3, 4)]
    imgs = build(pattern)
    out, mask, report, _ = run(imgs, preserve_first=False)
    assert 0 in keyframes_from(report)
    assert torch.allclose(out[0], imgs[0])
    check_consistency(pattern, out, mask, report)


def test_global_flags_off_do_nothing():
    pattern = [(1, 1), (2, 4), (3, 1)]
    imgs = build(pattern)
    out, mask, report, _ = run(imgs, preserve_global_first=False,
                               preserve_global_last=False,
                               insert_padding=False, align_keyframes=False)
    kfs = keyframes_from(report)
    assert 0 not in kfs and (out.shape[0] - 1) not in kfs
    # ...but frames outside every hold are still left untouched
    assert torch.allclose(out[0], imgs[0])
    assert torch.allclose(out[-1], imgs[-1])
    check_consistency(pattern, out, mask, report)


def test_padding_does_not_swallow_unique_frames():
    """Spong settings: padding on, alignment off, first-of-hold only.
    Two unique frames between holds must stay untouched."""
    pattern = [(1, 1), (2, 3), (3, 1), (4, 1), (5, 3)]
    imgs = build(pattern)
    out, mask, report, removal = run(
        imgs, preserve_last=False, preserve_global_last=False,
        min_gray_frames=8, align_keyframes=False,
    )
    check_consistency(pattern, out, mask, report)
    rows = mask_rows(mask)
    untouched = [i for i in range(out.shape[0])
                 if is_untouched_unique(out[i], pattern)]
    assert len(untouched) == 3, untouched  # scenes 1, 3 and 4
    assert all(rows[i] == 0 for i in untouched)
    # removing the padding restores the original frame count
    restored, _ = REMOVER.remove_frames(out, removal)
    assert restored.shape[0] == imgs.shape[0]
    assert torch.allclose(restored[0], imgs[0])


def test_padding_keeps_hold_last_frame():
    """With preserve_last on, the keyframe is the hold's real last frame,
    not the unique frame that precedes the next hold."""
    pattern = [(1, 4), (2, 1), (3, 4)]
    imgs = build(pattern)
    out, mask, report, removal = run(imgs, align_keyframes=False)
    check_consistency(pattern, out, mask, report)
    restored, _ = REMOVER.remove_frames(out, removal)
    rows = [int(mask[i].mean().round()) for i in range(out.shape[0])
            if str(i) not in removal.split(",")]
    # original layout: 0..3 hold A, 4 unique, 5..8 hold B
    assert rows == [0, 1, 1, 0, 0, 0, 1, 1, 0], rows
    assert torch.allclose(restored[3], imgs[3])
    assert torch.allclose(restored[4], imgs[4])


def test_alignment_has_no_stale_ranges():
    """After alignment no hold frame may leak through un-grayed, and no
    unique frame may be grayed."""
    pattern = [(1, 1), (2, 4), (3, 1), (4, 4)]
    imgs = build(pattern)
    out, mask, report, _ = run(imgs, insert_padding=False)
    check_consistency(pattern, out, mask, report)
    kfs = keyframes_from(report)
    assert all(k % 4 == 0 for k in kfs), kfs


def test_skip_second_to_last():
    pattern = [(1, 4), (2, 4), (3, 4)]
    imgs = build(pattern)
    _, _, report_plain, _ = run(imgs, preserve_last=False,
                                insert_padding=False, align_keyframes=False)
    out, mask, report, _ = run(imgs, preserve_last=False,
                               skip_second_to_last=True,
                               insert_padding=False, align_keyframes=False)
    plain = keyframes_from(report_plain)
    skipped = keyframes_from(report)
    assert skipped == [k for k in plain if k != plain[-2]]
    assert mask_rows(mask)[plain[-2]] == 1
    check_consistency(pattern, out, mask, report)


def test_global_last_ignores_trailing_padding():
    """min_sequence_length=1 pads after a unique last frame; the global
    last keyframe must be that original frame, never the padding."""
    pattern = [(1, 3), (2, 1)]
    imgs = build(pattern)
    out, mask, report, removal = run(imgs, min_sequence_length=1,
                                     preserve_first=False,
                                     preserve_last=False,
                                     align_keyframes=False)
    kfs = keyframes_from(report)
    removed = {int(x) for x in removal.split(",") if x}
    assert kfs[-1] not in removed
    assert torch.allclose(out[kfs[-1]], imgs[-1])
    check_consistency(pattern, out, mask, report, min_len=1)


def test_fuzz_invariants():
    rng = random.Random(0)
    for _ in range(200):
        pattern = [(i + 1, rng.choice([1, 1, 2, 2, 3, 4, 6]))
                   for i in range(rng.randint(2, 6))]
        imgs = build(pattern)
        kw = dict(
            preserve_first=rng.random() < 0.7,
            preserve_last=rng.random() < 0.7,
            preserve_global_first=rng.random() < 0.7,
            preserve_global_last=rng.random() < 0.7,
            skip_second=rng.random() < 0.2,
            skip_second_to_last=rng.random() < 0.2,
            min_sequence_length=rng.choice([1, 2, 2, 3]),
            min_gray_frames=rng.choice([0, 2, 4, 8]),
            insert_padding=rng.random() < 0.7,
            align_keyframes=rng.random() < 0.7,
            alignment_multiple=rng.choice([2, 4]),
        )
        out, mask, report, removal = run(imgs, **kw)
        check_consistency(pattern, out, mask, report,
                          min_len=kw["min_sequence_length"])
        kfs = keyframes_from(report)
        before_skip = int([l for l in report.splitlines()
                           if l.startswith("Total preserved frames before")][0]
                          .split(":")[1])
        # skip_second_to_last removes keyframe[-2], which is frame 0 when the
        # list holds exactly two entries; that is the documented behaviour.
        first_skipped = kw["skip_second_to_last"] and before_skip == 2
        if kw["preserve_global_first"] and not first_skipped:
            assert 0 in kfs, (pattern, kw, kfs)
            assert torch.allclose(out[0], imgs[0])
        if (kw["preserve_global_last"] and not kw["skip_second"]
                and not kw["skip_second_to_last"]):
            assert torch.allclose(out[kfs[-1]], imgs[-1]), (pattern, kw, kfs)
        if kw["align_keyframes"]:
            assert all(k % kw["alignment_multiple"] == 0 for k in kfs), \
                (pattern, kw, kfs)
        # removal indices are always valid and restore the pre-padding count
        restored, _ = REMOVER.remove_frames(out, removal)
        n_removed = len([x for x in removal.split(",") if x])
        assert restored.shape[0] == out.shape[0] - n_removed


def removed_set(removal):
    return {int(x) for x in removal.split(",") if x}


def test_next_valid_frame_count():
    f = AnimationDuplicateFrameProcessor.next_valid_frame_count
    assert [f(n, 4) for n in (1, 2, 5, 6, 8, 9, 20, 21, 22)] == \
        [5, 5, 5, 9, 9, 9, 21, 21, 25]
    assert f(10, 8) == 17 and f(17, 8) == 17 and f(3, 8) == 9


def test_pad_to_4n_plus_1_appends_removable_gray_tail():
    """20 frames, alignment off -> 21. The tail is gray, mask white, and
    listed in removal_indices; removing it restores the original count."""
    pattern = [(1, 5), (2, 5), (3, 5), (4, 5)]
    imgs = build(pattern)
    out, mask, report, removal = run(
        imgs, insert_padding=False, align_keyframes=False,
        pad_to_4n_plus_1=True,
    )
    assert out.shape[0] == 21 and mask.shape[0] == 21
    assert (out.shape[0] - 1) % 4 == 0
    assert mask_rows(mask)[-1] == 1
    assert not torch.allclose(out[-1], imgs[-1])          # it is gray
    assert removed_set(removal) == {20}
    kfs = keyframes_from(report)
    assert kfs[-1] == 19                                   # global last stays
    restored, _ = REMOVER.remove_frames(out, removal)
    assert restored.shape[0] == imgs.shape[0]
    assert torch.allclose(restored[-1], imgs[-1])
    check_consistency(pattern, out[:20], mask[:20], report)


def test_pad_to_4n_plus_1_after_alignment_and_padding():
    """With gray padding and alignment on, the final count is still 4n+1
    and every keyframe is still aligned."""
    pattern = [(1, 1), (2, 3), (3, 1), (4, 3)]
    imgs = build(pattern)
    out, mask, report, removal = run(imgs, pad_to_4n_plus_1=True)
    n = out.shape[0]
    assert (n - 1) % 4 == 0 and n >= 5
    assert mask.shape[0] == n
    kfs = keyframes_from(report)
    assert all(k % 4 == 0 for k in kfs), kfs
    assert max(kfs) < n
    # tail frames sit after the last keyframe and are all removable
    tail = [i for i in range(max(kfs) + 1, n)]
    assert set(tail) <= removed_set(removal)
    assert all(mask_rows(mask)[i] == 1 for i in tail)


def test_pad_to_4n_plus_1_noop_when_already_valid():
    pattern = [(1, 3), (2, 3), (3, 3)]   # 9 frames = 4*2+1
    imgs = build(pattern)
    a = run(imgs, insert_padding=False, align_keyframes=False)
    b = run(imgs, insert_padding=False, align_keyframes=False,
            pad_to_4n_plus_1=True)
    assert a[0].shape == b[0].shape and torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1]) and a[3] == b[3]


def test_pad_to_4n_plus_1_minimum_and_multiple():
    imgs = build([(1, 2), (2, 1)])       # 3 frames -> 5 for WAN, 9 for LTX
    out, mask, _, removal = run(imgs, insert_padding=False,
                                align_keyframes=False, pad_to_4n_plus_1=True)
    assert out.shape[0] == 5 and removed_set(removal) == {3, 4}
    out, mask, _, removal = run(imgs, insert_padding=False,
                                align_keyframes=False, pad_to_4n_plus_1=True,
                                frame_multiple=8)
    assert out.shape[0] == 9 and removed_set(removal) == set(range(3, 9))


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
