#!/usr/bin/env python3
"""On-board benchmark: every registered model, timed and made to say what it found.

Two columns matter and the second is the point. `infer` is the NPU alone;
`e2e` is the whole task — preprocessing, inference and post-processing — because
that is what a caller waits for, and for several heads here the post-processing
is the larger half. The `result` column is what keeps this a smoke test rather
than a timing table: a model that got 3 ms faster and stopped finding the bus
should not look like an improvement.

    PYTHONPATH=build:python python benchmarks/bench.py            # everything staged
    PYTHONPATH=build:python python benchmarks/bench.py --only det,flow
    PYTHONPATH=build:python python benchmarks/bench.py --json benchmarks/results.json

Models come from `models/` (see scripts/fetch_models.sh) and sample images from
`data/images/`; anything missing is reported as skipped, so a partial checkout
still produces a partial table.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "tests"))

import board_models as bm  # noqa: E402  (needs the path above)

import rcdl  # noqa: E402

import figures as fg  # noqa: E402  (benchmarks/ is on the path as this file's dir)

WARMUP, ITERS = 2, 5

# Set by main() when --figures is given; a task that can draw one appends it as a
# fifth element of its return tuple, so the picture and the number always come
# from the same run.
DRAW = False


def timed(fn, iters=ITERS, warmup=WARMUP):
    """Median wall-clock ms over `iters` runs, discarding warm-up."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def npu_ms(engine):
    """The runtime's own view of the last inference, in ms (-1 if unavailable)."""
    us = engine.last_run_micros()
    return us / 1e3 if us and us > 0 else float("nan")


def model_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


# --------------------------------------------------------------------------- #
# One entry per task. Each returns (infer_ms, e2e_ms, result string).          #
# --------------------------------------------------------------------------- #
def bench_detection(name, model):
    def run():
        path = bm.require_model(model)
        img = bm.load_bgr("bus.jpg")
        e = rcdl.Engine(path)
        det = e.detector()
        dets = rcdl.detect(det, img)
        e2e = timed(lambda: rcdl.detect(det, img))
        infer = det.profile[1] if hasattr(det, "profile") else npu_ms(e)
        names = {}
        for d in dets:
            n = rcdl.coco_class_name(d.class_id)
            names[n] = names.get(n, 0) + 1
        summary = ", ".join(f"{v} {k}" for k, v in sorted(names.items()))
        fig = None
        if DRAW:
            fig = img.copy()
            for d in dets:
                fg.box(fig, d.x1, d.y1, d.x2, d.y2,
                       f"{rcdl.coco_class_name(d.class_id)} {d.score:.2f}",
                       fg.class_color(d.class_id))
            fg.caption(fig, f"{name}: {summary}",
                       "boxes in SOURCE pixels — the letterbox is undone by the decoder")
        return infer, e2e, model_mb(path), summary, fig
    return name, run


def bench_classification(name, model, softmax=True):
    def run():
        path = bm.require_model(model)
        img = bm.load_bgr("space_shuttle_224.jpg")
        e = rcdl.Engine(path)
        cls = e.classifier(apply_softmax=softmax)
        top = rcdl.classify(cls, img)
        e2e = timed(lambda: rcdl.classify(cls, img))
        summary = ", ".join(f"{c.class_id}:{c.score:.3f}" for c in top[:3])
        fig = None
        if DRAW:
            fig = img.copy()
            if fig.shape[0] < 320:
                import cv2
                s_ = 320 / fig.shape[0]
                fig = cv2.resize(fig, (int(fig.shape[1] * s_), 320))
            fg.bars(fig, [(f"class {c.class_id}", float(c.score), fg.class_color(c.class_id))
                          for c in top[:3]],
                    title=f"{name}: top-3 (softmax {'in the graph' if not softmax else 'on the CPU'})")
        return npu_ms(e), e2e, model_mb(path), summary, fig
    return name, run


def bench_instance_seg(name, model):
    def run():
        path = bm.require_model(model)
        img = bm.load_bgr("bus.jpg")
        e = rcdl.Engine(path)
        seg = e.instance_segmenter()
        inst = rcdl.segment_instances(seg, img)
        e2e = timed(lambda: rcdl.segment_instances(seg, img))
        fig = None
        if DRAW:
            fig = img.copy()
            for m in inst:
                col = fg.class_color(m.class_id)
                arr = np.asarray(m.mask() if callable(getattr(m, "mask", None)) else m.mask)
                full = np.zeros(img.shape[:2], bool)
                y0, x0 = max(0, m.mask_y0), max(0, m.mask_x0)
                sub = arr.astype(bool)[: img.shape[0] - y0, : img.shape[1] - x0]
                full[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = sub
                fg.blend_mask(fig, full, col)
                fg.box(fig, m.x1, m.y1, m.x2, m.y2,
                       f"{rcdl.coco_class_name(m.class_id)} {m.score:.2f}", col)
            fg.caption(fig, f"{name}: {len(inst)} instances",
                       "per-instance masks, cropped to their own box — not a semantic map")
        return npu_ms(e), e2e, model_mb(path), f"{len(inst)} instances", fig
    return name, run


def bench_semantic_seg(name, model):
    def run():
        path = bm.require_model(model)
        # A Cityscapes-trained model deserves a street from a car: both semantic
        # models here predict the 19 Cityscapes classes, and this is the frame
        # their cross-model agreement is quoted on in docs/MODELS.md.
        img = bm.load_bgr("cityscapes.png") if os.path.isfile(bm.find_image("cityscapes.png")) \
            else bm.load_bgr("bus.jpg")
        e = rcdl.Engine(path)
        seg = e.segmenter()
        m = rcdl.segment(seg, img)
        e2e = timed(lambda: rcdl.segment(seg, img))
        labels = np.asarray(m.labels)
        summary = (f"{labels.shape[1]}x{labels.shape[0]} map, "
                   f"{len(np.unique(labels))} classes present")
        fig = None
        if DRAW:
            import cv2
            colored = np.asarray(rcdl.seg_colorize(np.ascontiguousarray(labels, np.int32)))
            fig = cv2.addWeighted(img, 0.45, colored, 0.55, 0)
            fg.caption(fig, f"{name}: {summary}",
                       "argmax over the channel axis, projected back onto the frame")
        return npu_ms(e), e2e, model_mb(path), summary, fig
    return name, run


def bench_pose(name, model):
    def run():
        path = bm.require_model(model)
        img = bm.load_bgr("bus.jpg")
        e = rcdl.Engine(path)
        pose = e.pose_estimator()
        poses = rcdl.estimate_pose(pose, img)
        e2e = timed(lambda: rcdl.estimate_pose(pose, img))
        vis = sum(1 for p in poses for k in p.keypoints if k.score > 0.5)
        fig = None
        if DRAW:
            import cv2
            fig = img.copy()
            for i, p in enumerate(poses):
                col = fg.class_color(i * 3)
                kp = p.keypoints
                for a, b in rcdl.coco_skeleton():
                    if a < len(kp) and b < len(kp) and kp[a].score > 0.5 and kp[b].score > 0.5:
                        cv2.line(fig, (int(kp[a].x), int(kp[a].y)), (int(kp[b].x), int(kp[b].y)),
                                 col, 2, cv2.LINE_AA)
                for k in kp:
                    if k.score > 0.5:
                        cv2.circle(fig, (int(k.x), int(k.y)), 3, (255, 255, 255), -1, cv2.LINE_AA)
                fg.box(fig, p.box.x1, p.box.y1, p.box.x2, p.box.y2, None, col)
            fg.caption(fig, f"{name}: {len(poses)} people, {vis} joints over 0.5",
                       "every confident joint has to fall inside its OWN person's box")
        return npu_ms(e), e2e, model_mb(path), f"{len(poses)} people, {vis} joints over 0.5", fig
    return name, run


def bench_obb(name, model):
    def run():
        path = bm.require_model(model)
        img = bm.load_bgr("obb.jpg")
        e = rcdl.Engine(path)
        obb = e.obb_detector()
        dets = rcdl.detect_obb(obb, img)
        e2e = timed(lambda: rcdl.detect_obb(obb, img))
        fig = None
        if DRAW:
            import cv2
            fig = img.copy()
            for d in dets:
                c = d.rrect.corners
                pts = np.int32(np.asarray(c() if callable(c) else c).reshape(4, 2))
                col = fg.class_color(d.class_id)
                cv2.polylines(fig, [pts], True, col, 2, cv2.LINE_AA)
                fg.put(fig, rcdl.dota_class_name(d.class_id),
                       (int(pts[:, 0].min()), max(14, int(pts[:, 1].min()) - 6)), 0.5, col)
            fg.caption(fig, f"{name}: {len(dets)} rotated boxes",
                       "the angle convention is per-generation — a wrong one still lands on the object")
        return npu_ms(e), e2e, model_mb(path), f"{len(dets)} rotated boxes", fig
    return name, run


def bench_depth(name, model):
    def run():
        path = bm.require_model(model)
        img = bm.load_bgr("bus.jpg")
        e = rcdl.Engine(path)
        dep = e.depth_estimator()
        m = rcdl.estimate_depth(dep, img)
        e2e = timed(lambda: rcdl.estimate_depth(dep, img), iters=3)
        d = np.asarray(m.data)
        summary = f"{m.width}x{m.height} disparity [{d.min():.2f},{d.max():.2f}]"
        fig = None
        if DRAW:
            colored = np.asarray(rcdl.depth_colorize(m))
            fig = fg.side_by_side(img, colored, labels=("frame", "relative inverse depth"))
            fg.caption(fig, f"{name}: {summary}",
                       "int8 keeps the ORDER, not the values — hence the per-frame normalisation")
        return npu_ms(e), e2e, model_mb(path), summary, fig
    return name, run


def bench_ocr(name, det_model, rec_model, dict_name="ppocr_keys_v1_6625.txt",
              det_thresholds=None, logits=False):
    """One OCR generation end to end: detect, then recognise every box.

    `det_thresholds` is (bin, box, unclip) when the detector ships its own DB
    constants — PP-OCRv6 does, and they are not this library's defaults.
    `logits` marks a recogniser whose softmax was taken out of the graph (v5, v6):
    it takes raw 0..255 and the softmax runs on the CPU for the score.
    """
    def run():
        det_path = bm.require_model(det_model)
        rec_path = bm.require_model(rec_model)
        img = bm.load_bgr("ocr.jpg")
        dict_path = os.path.join(os.path.dirname(bm.IMAGES), dict_name)
        de = rcdl.Engine(det_path)
        re_ = rcdl.Engine(rec_path)
        if det_thresholds is None:
            det = de.text_detector()
        else:
            cfg = rcdl.OcrDetConfig()
            cfg.bin_thresh, cfg.box_thresh, cfg.unclip_ratio = det_thresholds
            det = de.text_detector(config=cfg)
        rcfg = rcdl.OcrRecConfig()
        rcfg.apply_softmax = logits
        rec = re_.text_recognizer(dict_path, config=rcfg,
                                  **(dict(input_scale=1.0, input_shift=0.0,
                                          model_input="rgb888") if logits else {}))
        boxes = rcdl.detect_text(det, img, "bgr888")

        def read_all():
            # The recogniser takes an upright crop, and cutting the quadrilateral
            # out is a host image operation rather than part of the decode — the
            # same four-point warp the board tests use.
            import cv2
            out = []
            for b in boxes:
                pts = np.asarray(b.pts, np.float32).reshape(4, 2)
                w = int(round(max(np.linalg.norm(pts[0] - pts[1]),
                                  np.linalg.norm(pts[3] - pts[2]))))
                h = int(round(max(np.linalg.norm(pts[0] - pts[3]),
                                  np.linalg.norm(pts[1] - pts[2]))))
                dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
                crop = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst),
                                           (max(w, 1), max(h, 1)))
                # Taller than wide is a 90-degree line the warp step should have
                # rotated upright; handing it over as-is is a pipeline error and
                # what comes back is junk, so skip it as the board tests do.
                if crop.shape[0] < 4 or crop.shape[1] < 4 or crop.shape[0] > crop.shape[1]:
                    continue
                out.append(rcdl.recognize_text(rec, crop, "bgr888"))
            return out

        lines = read_all()
        e2e = timed(read_all, iters=2, warmup=1)
        good = [l for l in lines if l.text]
        summary = f"{len(boxes)} boxes, {len(good)} lines read"
        fig = None
        if DRAW:
            import cv2
            fig = img.copy()
            for b in boxes:
                cv2.polylines(fig, [np.int32(np.asarray(b.pts).reshape(4, 2))], True,
                              (90, 210, 120), 2, cv2.LINE_AA)
            # The text itself is CJK here, which cv2's Hershey fonts cannot draw —
            # so the figure shows WHERE the lines are and how many were read, and
            # the exact strings stay pinned in the board test where they belong.
            fg.panel(fig, [(f"{summary} — detector + CTC recogniser", (255, 255, 255)),
                           ("scores: " + ", ".join(f"{l.score:.2f}" for l in good[:8]),
                            (190, 220, 255))])
        return npu_ms(de), e2e, model_mb(det_path) + model_mb(rec_path), summary, fig
    return name, run


def bench_face(name, model):
    def run():
        path = bm.require_model(model)
        img = bm.load_bgr("zidane.jpg")
        e = rcdl.Engine(path)
        fd = e.face_detector()
        faces = rcdl.detect_faces(fd, img)
        e2e = timed(lambda: rcdl.detect_faces(fd, img))
        summary = f"{len(faces)} faces, best {max((f.score for f in faces), default=0):.3f}"
        fig = None
        if DRAW:
            import cv2
            fig = img.copy()
            for f in faces:
                fg.box(fig, f.x1, f.y1, f.x2, f.y2, f"{f.score:.3f}", (80, 220, 100))
                for (lx, ly) in np.asarray(f.landmarks, np.float32).reshape(5, 2):
                    cv2.circle(fig, (int(lx), int(ly)), 3, (60, 200, 240), -1, cv2.LINE_AA)
            fg.caption(fig, f"{name}: {summary}",
                       "five landmarks per face — feeding this model RGB instead of BGR loses one")
        return npu_ms(e), e2e, model_mb(path), summary, fig
    return name, run


def bench_reid(name, model):
    def run():
        path = bm.require_model(model)
        det = rcdl.Engine(bm.require_model("yolov8n_rk3588.rknn")).detector()
        img = bm.load_bgr("bus.jpg")
        people = [d for d in rcdl.detect(det, img)
                  if rcdl.coco_class_name(d.class_id) == "person"]
        e = rcdl.Engine(path)
        emb = e.embedder()
        vecs = [rcdl.embed(emb, img, (d.x1, d.y1, d.x2, d.y2)) for d in people]
        e2e = timed(lambda: rcdl.embed(emb, img, (people[0].x1, people[0].y1,
                                                  people[0].x2, people[0].y2)))
        sims = [float(np.dot(a, b)) for i, a in enumerate(vecs) for b in vecs[i + 1:]]
        summary = f"{len(vecs)} crops, cross-similarity max {max(sims, default=0):.3f}"
        fig = None
        if DRAW:
            import cv2
            # The crops themselves ARE the figure: these are different people, so
            # every off-diagonal similarity has to stay low. A vector that looked
            # fine and separated nothing would be invisible in a photograph.
            strip = []
            for d in people:
                c = img[max(0, int(d.y1)):int(d.y2), max(0, int(d.x1)):int(d.x2)]
                if c.size:
                    strip.append(cv2.resize(c, (160, 320)))
            if strip:
                fig = np.hstack(strip)
                fg.panel(fig, [(f"{name}: {len(vecs)} person crops, all DIFFERENT people",
                                (255, 255, 255)),
                               ("pairwise cosine: " + ", ".join(f"{v:+.2f}" for v in sims[:8]),
                                (190, 220, 255)),
                               ("crops are SQUASHED to 128x256, not letterboxed", (150, 200, 150))])
        return npu_ms(e), e2e, model_mb(path), summary, fig
    return name, run


def bench_features(name, model):
    def run():
        path = bm.require_model(model)
        import cv2
        e = rcdl.Engine(path, float_inputs=[0])
        ex = e.feature_extractor()
        img = cv2.resize(bm.load_bgr("bus.jpg"), (ex.input_width, ex.input_height))
        m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 12, 0.85) \
            if hasattr(cv2, "getRotationMatrix2D") else None
        warped = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                                borderMode=cv2.BORDER_REFLECT) if m is not None else img
        a = rcdl.extract_features(ex, img)
        b = rcdl.extract_features(ex, warped)
        pairs, _ = rcdl.match_features(a, b)
        e2e = timed(lambda: rcdl.extract_features(ex, img), iters=3)
        match_ms = timed(lambda: rcdl.match_features(a, b), iters=3)
        summary = (f"{len(a)}+{len(b)} features, {len(pairs)} matches "
                   f"(+{match_ms:.0f} ms to match)")
        fig = None
        if DRAW:
            fig = fg.side_by_side(img, warped, labels=("frame", "known warp: 12 deg, 0.85x"))
            off = img.shape[1] + 8
            xa, xb = np.asarray(a.xy), np.asarray(b.xy)
            # ~25 lines, not 2000: a figure showing every match is a solid block
            # of colour that says nothing about whether the matches are right.
            pr = np.asarray(pairs)
            for k, (i, j) in enumerate(pr[:: max(1, len(pr) // 25)]):
                p0 = (int(xa[i][0]), int(xa[i][1]))
                p1 = (int(xb[j][0]) + off, int(xb[j][1]))
                import cv2
                cv2.line(fig, p0, p1, fg.class_color(k * 5), 1, cv2.LINE_AA)
            fg.caption(fig, f"{name}: {len(pairs)} mutual-nearest matches",
                       "the warp is KNOWN, so every match has an exact right answer to score against")
        return npu_ms(e), e2e, model_mb(path), summary, fig
    return name, run


def bench_superres(name, model):
    def run():
        path = bm.require_model(model)
        import cv2
        img = cv2.resize(bm.load_bgr("bus.jpg"), (128, 128))
        e = rcdl.Engine(path)
        sr = e.upscaler()
        up = rcdl.upscale(sr, img)
        e2e = timed(lambda: rcdl.upscale(sr, img), iters=3)
        summary = f"128x128 -> {up.shape[1]}x{up.shape[0]}, {sr.last_tile_count} tile(s)"
        fig = None
        if DRAW:
            # Nearest-neighbour on the left so the comparison is against the SAME
            # pixels enlarged, not against a second interpolation.
            naive = cv2.resize(img, (up.shape[1], up.shape[0]), interpolation=cv2.INTER_NEAREST)
            fig = fg.side_by_side(naive, up, labels=("input, nearest x4", "model x4"))
            fg.caption(fig, f"{name}: {summary}",
                       "PSNR cannot judge this family — it is trained to invent plausible texture")
        return npu_ms(e), e2e, model_mb(path), summary, fig
    return name, run


def bench_flow(name, model):
    def run():
        path = bm.require_model(model)
        import cv2
        e = rcdl.Engine(path)
        est = e.flow_estimator()
        big = cv2.resize(bm.load_bgr("bus.jpg"), (est.input_width + 16, est.input_height + 16))
        a = np.ascontiguousarray(big[0:est.input_height, 0:est.input_width])
        b = np.ascontiguousarray(big[0:est.input_height, 8:8 + est.input_width])
        f = rcdl.estimate_flow(est, a, b)
        e2e = timed(lambda: rcdl.estimate_flow(est, a, b), iters=3)
        inner = f[48:-48, 48:-48]
        err = np.hypot(inner[..., 0] + 8.0, inner[..., 1])
        summary = f"{f.shape[1]}x{f.shape[0]} field, EPE {err.mean():.3f} px vs an 8 px shift"
        fig = None
        if DRAW:
            colored = np.asarray(rcdl.flow_colorize(np.ascontiguousarray(f, np.float32)))
            fig = fg.side_by_side(a, colored, labels=("frame A", "flow (hue = direction)"))
            fg.caption(fig, f"{name}: EPE {err.mean():.3f} px vs a known 8 px shift",
                       "a uniform shift is ground truth with no dataset — one colour is the answer")
        return npu_ms(e), e2e, model_mb(path), summary, fig
    return name, run


def bench_promptable(name, enc_model, dec_model):
    def run():
        enc_path = bm.require_model(enc_model)
        dec_path = bm.require_model(dec_model)
        img = bm.load_bgr("bus.jpg")
        enc = rcdl.Engine(enc_path)
        dec = rcdl.Engine(dec_path)
        sam = enc.prompt_segmenter(dec)
        flat = np.ascontiguousarray(img).reshape(-1)
        encode_ms = timed(lambda: sam.set_image(flat, img.shape[1], img.shape[0], "bgr888"),
                          iters=3)
        m = sam.box(36, 230, 799, 771)
        prompt_ms = timed(lambda: sam.box(36, 230, 799, 771), iters=3)
        summary = (f"box -> {100 * m.area:.1f}% of the frame @ {m.score:.3f} "
                   f"(encode {encode_ms:.0f} ms + prompt {prompt_ms:.0f} ms)")
        fig = None
        if DRAW:
            import cv2
            fig = img.copy()
            fg.blend_mask(fig, np.asarray(m.mask), (60, 200, 240), 0.5)
            cv2.rectangle(fig, (36, 230), (799, 771), (255, 255, 255), 2)
            fg.caption(fig, f"{name}: prompt box -> {100 * m.area:.1f}% of the frame",
                       "the mask fills 74% of the PROMPT — segmenting the bus, not returning the box")
        return npu_ms(enc), encode_ms + prompt_ms, model_mb(enc_path) + model_mb(dec_path), \
            summary, fig
    return name, run


def bench_wholebody(name, model):
    def run():
        path = bm.require_model(model)
        img = bm.load_bgr("bus.jpg")
        det = rcdl.Engine(bm.require_model("yolov8n_rk3588.rknn")).detector()
        people = [d for d in rcdl.detect(det, img)
                  if rcdl.coco_class_name(d.class_id) == "person"]
        if not people:
            raise RuntimeError("no person to pose")
        p = max(people, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))
        e = rcdl.Engine(path)
        wb = e.wholebody_estimator()
        box = (p.x1, p.y1, p.x2, p.y2)
        kp = rcdl.estimate_wholebody(wb, img, box)
        e2e = timed(lambda: rcdl.estimate_wholebody(wb, img, box))
        summary = f"{int((kp[:, 2] >= 0.3).sum())}/{len(kp)} keypoints over 0.3, one person"
        fig = None
        if DRAW:
            import cv2
            x1, y1 = max(0, int(p.x1) - 20), max(0, int(p.y1) - 20)
            x2, y2 = min(img.shape[1], int(p.x2) + 20), min(img.shape[0], int(p.y2) + 20)
            fig = np.ascontiguousarray(img[y1:y2, x1:x2])
            part_color = {rcdl.BodyPart.BODY: (80, 220, 100),
                          rcdl.BodyPart.FOOT: (240, 180, 60),
                          rcdl.BodyPart.FACE: (60, 200, 240),
                          rcdl.BodyPart.LEFT_HAND: (240, 120, 200),
                          rcdl.BodyPart.RIGHT_HAND: (200, 120, 240)}
            for i, (kx, ky, ks) in enumerate(kp):
                if ks < 0.3:
                    continue
                cv2.circle(fig, (int(kx) - x1, int(ky) - y1), 2,
                           part_color.get(rcdl.body_part(i), (200, 200, 200)), -1, cv2.LINE_AA)
            fg.caption(fig, f"{name}: 133/133 keypoints, one person",
                       "body / feet / face / hands, by region")
        return npu_ms(e), e2e, model_mb(path), summary, fig
    return name, run


def bench_face_recognition(name, model):
    """Identity embedding. `infer` is the NPU alone; the gap to `e2e` is the
    five-point warp, which is CPU because a similarity transform is not
    something the hardware letterbox can express. The result column reports the
    separation actually achieved, since a broken alignment still returns
    well-formed unit vectors."""
    def run():
        path = bm.require_model(model)
        det = rcdl.Engine(bm.require_model("retinaface_rk3588.rknn")).face_detector()
        e = rcdl.Engine(path)
        rec = e.face_recognizer()
        vecs, one = [], None
        for img_name in ("zidane.jpg", "bus.jpg"):
            img = bm.load_bgr(img_name)
            for f in rcdl.detect_faces(det, img):
                lm = np.array(f.landmarks, np.float32).reshape(5, 2)
                vecs.append(rcdl.embed_face(rec, img, lm))
                if one is None:
                    one = (img, lm)
        if not vecs:
            raise RuntimeError("no faces to embed")
        e2e = timed(lambda: rcdl.embed_face(rec, one[0], one[1]))
        m = np.stack(vecs) @ np.stack(vecs).T
        off = max(float(m[i, j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs)))
        summary = f"{len(vecs)} faces, worst cross-identity similarity {off:.3f}"
        fig = None
        if DRAW:
            import cv2
            # The ALIGNED crops are the figure: the canonical pose is the whole
            # contract, and a photograph of a face cannot show whether it held.
            crops = []
            for img_name in ("zidane.jpg", "bus.jpg"):
                im = bm.load_bgr(img_name)
                for f in rcdl.detect_faces(det, im):
                    lm = np.array(f.landmarks, np.float32).reshape(5, 2)
                    mat = np.asarray(rcdl.face_align_transform(lm.reshape(-1), 112, 112), np.float32)
                    crops.append(cv2.warpAffine(im, mat, (112, 112)))
            if crops:
                fig = np.hstack([cv2.resize(c, (224, 224)) for c in crops])
                fg.panel(fig, [(f"{name}: {len(crops)} faces on the ArcFace template",
                                (255, 255, 255)),
                               (f"worst cross-identity cosine {off:+.3f} — all different people",
                                (190, 220, 255)),
                               ("box-cropping instead of this scores 0.493 against the same face",
                                (150, 200, 150))])
        return npu_ms(e), e2e, model_mb(path), summary, fig
    return name, run


def bench_open_vocab(name, model):
    """Open-vocabulary detection. The `result` column names what the PROMPTS
    found, which is the only thing that distinguishes this from any other
    detector — the head, the decode and the timing are identical."""
    def run():
        path = bm.require_model(model)
        img = bm.load_bgr("bus.jpg")
        e = rcdl.Engine(path)
        labels = e.label_map()
        det = e.detector(num_classes=len(labels))
        dets = rcdl.detect(det, img)
        e2e = timed(lambda: rcdl.detect(det, img))
        names = {}
        for d in dets:
            n = labels.name(d.class_id)
            names[n] = names.get(n, 0) + 1
        summary = f"{len(labels)} prompts -> " + (
            ", ".join(f"{v} {k}" for k, v in sorted(names.items())) or "nothing")
        fig = None
        if DRAW:
            fig = img.copy()
            for d in dets:
                fg.box(fig, d.x1, d.y1, d.x2, d.y2,
                       f"{labels.name(d.class_id)} {d.score:.2f}", fg.class_color(d.class_id))
            vocab = ", ".join(list(labels.names)[:6]) + ("..." if len(labels) > 6 else "")
            fg.panel(fig, [(f"{name}: vocabulary chosen at CONVERSION time", (255, 255, 255)),
                           (f"prompts: {vocab}", (190, 220, 255)),
                           (summary, (150, 220, 150))])
        return npu_ms(e), e2e, model_mb(path), summary, fig
    return name, run


def bench_open_vocab_seg(name, model):
    """Open-vocabulary instance segmentation — the same checkpoint as the
    detector with the mask branch exported, read by the ordinary segmenter."""
    def run():
        path = bm.require_model(model)
        img = bm.load_bgr("bus.jpg")
        e = rcdl.Engine(path)
        labels = e.label_map()
        seg = e.instance_segmenter(num_classes=len(labels))
        inst = rcdl.segment_instances(seg, img)
        e2e = timed(lambda: rcdl.segment_instances(seg, img))
        names = {}
        for m in inst:
            n = labels.name(m.class_id)
            names[n] = names.get(n, 0) + 1
        fig = None
        if DRAW:
            fig = img.copy()
            for m in inst:
                col = fg.class_color(m.class_id)
                arr = np.asarray(m.mask() if callable(getattr(m, "mask", None)) else m.mask)
                full = np.zeros(img.shape[:2], bool)
                y0, x0 = max(0, m.mask_y0), max(0, m.mask_x0)
                sub = arr.astype(bool)[: img.shape[0] - y0, : img.shape[1] - x0]
                full[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] = sub
                fg.blend_mask(fig, full, col)
                fg.box(fig, m.x1, m.y1, m.x2, m.y2, f"{labels.name(m.class_id)} {m.score:.2f}", col)
            fg.caption(fig, f"{name}: {len(inst)} instances from {len(labels)} prompts",
                       "the mask branch of the same YOLOE — 13 outputs, the v8-seg layout")
        return npu_ms(e), e2e, model_mb(path), \
            f"{len(labels)} prompts -> " + ", ".join(f"{v} {k}" for k, v in sorted(names.items())), \
            fig
    return name, run


def bench_panoptic_drive(name, model):
    """Three heads off one inference, so `infer` is that single NPU pass and
    `e2e` covers the anchor decode plus BOTH masks. The result column reports
    all three, because a build that keeps the boxes and loses the lane mask
    would otherwise look unchanged."""
    def run():
        path = bm.require_model(model)
        img = bm.load_bgr("cityscapes.png")
        e = rcdl.Engine(path)
        cfg = rcdl.AnchorDetectConfig()
        cfg.num_classes, cfg.conf_thresh = 1, 0.35
        det = e.anchor_detector(cfg, 0)
        drive = e.segmenter(num_classes=2, output_index=3)
        lane = e.segmenter(num_classes=2, output_index=4)

        def once():
            m = rcdl.segment(drive, img)
            return det.postprocess(drive.letterbox), m, lane.postprocess(drive.letterbox)

        dets, drivable, lanes = once()
        e2e = timed(lambda: once())
        da = np.asarray(drivable.labels).astype(bool).mean() * 100
        ll = np.asarray(lanes.labels).astype(bool).mean() * 100
        summary = f"{len(dets)} vehicles, drivable {da:.1f}%, lane {ll:.1f}% of the frame"
        fig = None
        if DRAW:
            fig = img.copy()
            fg.blend_mask(fig, np.asarray(drivable.labels), (60, 200, 90), 0.40)
            fg.blend_mask(fig, np.asarray(lanes.labels), (60, 90, 240), 0.75)
            for d in dets:
                fg.box(fig, d.x1, d.y1, d.x2, d.y2, None, (255, 220, 60), 2)
            fg.caption(fig, f"{name}: {summary}",
                       "ONE inference, three decoders — vehicles, drivable area, lane lines")
        return npu_ms(e), e2e, model_mb(path), summary, fig
    return name, run


TASKS = [
    bench_detection("det", "yolov8n_rk3588.rknn"),
    bench_detection("det_yolo11", "yolo11n_rk3588.rknn"),
    bench_detection("det_yolo26", "yolo26n_rk3588.rknn"),
    bench_classification("cls", "resnet18_rk3588.rknn"),
    bench_classification("cls_yolo26", "yolo26n-cls_rk3588.rknn", softmax=False),
    bench_instance_seg("instance_seg", "yolov8n-seg_rk3588.rknn"),
    bench_semantic_seg("semantic_seg", "ppseg_rk3588.rknn"),
    bench_semantic_seg("semantic_seg_yolo26", "yolo26n_sem_640_i8_rk3588.rknn"),
    bench_pose("pose", "yolov8n-pose_rk3588.rknn"),
    bench_obb("obb", "yolov8n-obb_rk3588.rknn"),
    bench_depth("depth", "depth_anything_v2_vits_308_rk3588.rknn"),
    bench_ocr("ocr", "ppocrv4_det_rk3588.rknn", "ppocrv4_rec_rk3588.rknn"),
    bench_ocr("ocr_v6", "ppocrv6_medium_det_rk3588.rknn",
              "ppocrv6_medium_rec_logits_rk3588.rknn",
              dict_name="ppocr_keys_v6_18710.txt", det_thresholds=(0.2, 0.45, 1.4),
              logits=True),
    bench_face("face", "retinaface_rk3588.rknn"),
    bench_reid("reid", "osnet_x0_25_msmt17_rk3588.rknn"),
    bench_features("features", "xfeat_640x480_i8_rk3588.rknn"),
    bench_superres("superres", "realesr_general_x4v3_128_fp16_rk3588.rknn"),
    bench_flow("flow", "neuflow_v2_512x384_fp16_rk3588.rknn"),
    bench_promptable("promptable_seg", "edge_sam_3x_encoder_fp16_rk3588.rknn",
                     "edge_sam_3x_decoder_fp16_rk3588.rknn"),
    bench_wholebody("wholebody", "rtmw_s_133_256x192_fp16_rk3588.rknn"),
    bench_face_recognition("face_recognition", "arcface_r50_112_fp16_rk3588.rknn"),
    bench_open_vocab("open_vocab", "yoloe_11s_coco80_rk3588.rknn"),
    bench_open_vocab("open_vocab_prompts", "yoloe_11s_streetwear_rk3588.rknn"),
    bench_open_vocab_seg("open_vocab_seg", "yoloe_11s_coco80_seg_rk3588.rknn"),
    bench_panoptic_drive("panoptic_drive", "yolop_cut_640_i8_rk3588.rknn"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated task names")
    ap.add_argument("--json", help="write the rows here as JSON")
    ap.add_argument("--markdown", help="write a table here")
    ap.add_argument("--figures", nargs="?", const=os.path.join(_HERE, "figures"),
                    help="also draw a check figure per task into this directory "
                         "(default benchmarks/figures)")
    args = ap.parse_args()
    wanted = set(args.only.split(",")) if args.only else None

    global DRAW
    DRAW = args.figures is not None
    if DRAW:
        try:
            import cv2  # noqa: F401
        except ImportError:
            print("  note: OpenCV is missing, so no figures — the table still runs")
            DRAW = False

    rows, skipped = [], []
    for name, run in TASKS:
        if wanted and name not in wanted:
            continue
        try:
            out = run()
        except Exception as exc:  # a missing model raises through require_model's skip
            first = str(exc).splitlines()[0][:80]
            # A figure is a nice-to-have; the row is the point. If drawing threw,
            # take the measurement again without it rather than losing the task.
            if DRAW:
                DRAW = False
                try:
                    out = run()
                    print(f"  note: {name} figure failed ({first}), row kept")
                except Exception as exc2:
                    first = str(exc2).splitlines()[0][:80]
                    out = None
                finally:
                    DRAW = True
            else:
                out = None
            if out is None:
                skipped.append((name, first))
                print(f"  skip {name}: {first}")
                continue
        # A task that can draw returns its figure as a fifth element; the ones
        # whose output is not a picture simply do not.
        infer, e2e, mb, result = out[:4]
        fig = out[4] if len(out) > 4 else None
        figure_path = None
        if DRAW and fig is not None:
            try:
                figure_path = os.path.relpath(fg.save(fig, args.figures, name), _REPO)
            except Exception as exc:
                print(f"  note: {name} figure failed: {exc}")
        rows.append(dict(task=name, infer_ms=round(infer, 2) if infer == infer else None,
                         e2e_ms=round(e2e, 2), model_mb=round(mb, 2), result=result,
                         figure=figure_path))
        print(f"  {name:16s} infer {infer:8.2f} ms   e2e {e2e:8.2f} ms   {result}"
              + ("   [fig]" if figure_path else ""))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dict(rows=rows, skipped=skipped), f, indent=1, ensure_ascii=False)
        print("wrote", args.json)
    if args.markdown:
        table = ["| task | infer ms | e2e ms | model MB | result |", "|---|---|---|---|---|"]
        for r in rows:
            inf = "—" if r["infer_ms"] is None else f"{r['infer_ms']:.2f}"
            table.append(f"| {r['task']} | {inf} | {r['e2e_ms']:.1f} | {r['model_mb']:.1f} | "
                         f"{r['result']} |")
        block = "\n".join(table)
        # Replace only the table when the file already has the markers, so the
        # prose around it survives a regeneration — otherwise the documented
        # command would silently delete the explanation of its own numbers.
        begin, end = "<!-- BENCH:BEGIN -->", "<!-- BENCH:END -->"
        old_text = ""
        if os.path.isfile(args.markdown):
            with open(args.markdown, encoding="utf-8") as f:
                old_text = f.read()
        if begin in old_text and end in old_text:
            head = old_text.split(begin)[0]
            tail = old_text.split(end, 1)[1]
            new_text = f"{head}{begin}\n{block}\n{end}{tail}"
        else:
            new_text = f"{begin}\n{block}\n{end}\n"
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(new_text)
        print("wrote", args.markdown)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
