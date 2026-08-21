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

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "tests"))

import board_models as bm  # noqa: E402  (needs the path above)

import rcdl  # noqa: E402

WARMUP, ITERS = 2, 5


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
        return infer, e2e, model_mb(path), ", ".join(f"{v} {k}" for k, v in sorted(names.items()))
    return name, run


def bench_classification(name, model, softmax=True):
    def run():
        path = bm.require_model(model)
        img = bm.load_bgr("space_shuttle_224.jpg")
        e = rcdl.Engine(path)
        cls = e.classifier(apply_softmax=softmax)
        top = rcdl.classify(cls, img)
        e2e = timed(lambda: rcdl.classify(cls, img))
        return npu_ms(e), e2e, model_mb(path), \
            ", ".join(f"{c.class_id}:{c.score:.3f}" for c in top[:3])
    return name, run


def bench_instance_seg(name, model):
    def run():
        path = bm.require_model(model)
        img = bm.load_bgr("bus.jpg")
        e = rcdl.Engine(path)
        seg = e.instance_segmenter()
        inst = rcdl.segment_instances(seg, img)
        e2e = timed(lambda: rcdl.segment_instances(seg, img))
        return npu_ms(e), e2e, model_mb(path), f"{len(inst)} instances"
    return name, run


def bench_semantic_seg(name, model):
    def run():
        path = bm.require_model(model)
        img = bm.load_bgr("street.jpg") if os.path.isfile(bm.find_image("street.jpg")) \
            else bm.load_bgr("bus.jpg")
        e = rcdl.Engine(path)
        seg = e.segmenter()
        m = rcdl.segment(seg, img)
        e2e = timed(lambda: rcdl.segment(seg, img))
        labels = np.asarray(m.labels)
        return npu_ms(e), e2e, model_mb(path), \
            f"{labels.shape[1]}x{labels.shape[0]} map, {len(np.unique(labels))} classes present"
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
        return npu_ms(e), e2e, model_mb(path), f"{len(poses)} people, {vis} joints over 0.5"
    return name, run


def bench_obb(name, model):
    def run():
        path = bm.require_model(model)
        img = bm.load_bgr("obb.jpg")
        e = rcdl.Engine(path)
        obb = e.obb_detector()
        dets = rcdl.detect_obb(obb, img)
        e2e = timed(lambda: rcdl.detect_obb(obb, img))
        return npu_ms(e), e2e, model_mb(path), f"{len(dets)} rotated boxes"
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
        return npu_ms(e), e2e, model_mb(path), \
            f"{m.width}x{m.height} disparity [{d.min():.2f},{d.max():.2f}]"
    return name, run


def bench_ocr(name, det_model, rec_model):
    def run():
        det_path = bm.require_model(det_model)
        rec_path = bm.require_model(rec_model)
        img = bm.load_bgr("ocr.jpg")
        dict_path = os.path.join(os.path.dirname(bm.IMAGES), "ppocr_keys_v1_6625.txt")
        de = rcdl.Engine(det_path)
        re_ = rcdl.Engine(rec_path)
        det = de.text_detector()
        rec = re_.text_recognizer(dict_path)
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
                if crop.shape[0] < 4 or crop.shape[1] < 4:
                    continue
                out.append(rcdl.recognize_text(rec, crop, "bgr888"))
            return out

        lines = read_all()
        e2e = timed(read_all, iters=2, warmup=1)
        good = [l for l in lines if l.text]
        return npu_ms(de), e2e, model_mb(det_path) + model_mb(rec_path), \
            f"{len(boxes)} boxes, {len(good)} lines read"
    return name, run


def bench_face(name, model):
    def run():
        path = bm.require_model(model)
        img = bm.load_bgr("zidane.jpg")
        e = rcdl.Engine(path)
        fd = e.face_detector()
        faces = rcdl.detect_faces(fd, img)
        e2e = timed(lambda: rcdl.detect_faces(fd, img))
        return npu_ms(e), e2e, model_mb(path), \
            f"{len(faces)} faces, best {max((f.score for f in faces), default=0):.3f}"
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
        return npu_ms(e), e2e, model_mb(path), \
            f"{len(vecs)} crops, cross-similarity max {max(sims, default=0):.3f}"
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
        return npu_ms(e), e2e, model_mb(path), \
            f"{len(a)}+{len(b)} features, {len(pairs)} matches (+{match_ms:.0f} ms to match)"
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
        return npu_ms(e), e2e, model_mb(path), \
            f"128x128 -> {up.shape[1]}x{up.shape[0]}, {sr.last_tile_count} tile(s)"
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
        return npu_ms(e), e2e, model_mb(path), \
            f"{f.shape[1]}x{f.shape[0]} field, EPE {err.mean():.3f} px vs an 8 px shift"
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
        return npu_ms(enc), encode_ms + prompt_ms, model_mb(enc_path) + model_mb(dec_path), \
            f"box -> {100 * m.area:.1f}% of the frame @ {m.score:.3f} " \
            f"(encode {encode_ms:.0f} ms + prompt {prompt_ms:.0f} ms)"
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
        return npu_ms(e), e2e, model_mb(path), \
            f"{int((kp[:, 2] >= 0.3).sum())}/{len(kp)} keypoints over 0.3, one person"
    return name, run


TASKS = [
    bench_detection("det", "yolov8n_rk3588.rknn"),
    bench_detection("det_yolo11", "yolo11n_rk3588.rknn"),
    bench_detection("det_yolo26", "yolo26n_rk3588.rknn"),
    bench_classification("cls", "resnet18_rk3588.rknn"),
    bench_classification("cls_yolo26", "yolo26n-cls_rk3588.rknn", softmax=False),
    bench_instance_seg("instance_seg", "yolov8n-seg_rk3588.rknn"),
    bench_semantic_seg("semantic_seg", "ppseg_rk3588.rknn"),
    bench_pose("pose", "yolov8n-pose_rk3588.rknn"),
    bench_obb("obb", "yolov8n-obb_rk3588.rknn"),
    bench_depth("depth", "depth_anything_v2_vits_308_rk3588.rknn"),
    bench_ocr("ocr", "ppocrv4_det_rk3588.rknn", "ppocrv4_rec_rk3588.rknn"),
    bench_face("face", "retinaface_rk3588.rknn"),
    bench_reid("reid", "osnet_x0_25_msmt17_rk3588.rknn"),
    bench_features("features", "xfeat_640x480_i8_rk3588.rknn"),
    bench_superres("superres", "realesr_general_x4v3_128_fp16_rk3588.rknn"),
    bench_flow("flow", "neuflow_v2_512x384_fp16_rk3588.rknn"),
    bench_promptable("promptable_seg", "edge_sam_3x_encoder_fp16_rk3588.rknn",
                     "edge_sam_3x_decoder_fp16_rk3588.rknn"),
    bench_wholebody("wholebody", "rtmw_s_133_256x192_fp16_rk3588.rknn"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated task names")
    ap.add_argument("--json", help="write the rows here as JSON")
    ap.add_argument("--markdown", help="write a table here")
    args = ap.parse_args()
    wanted = set(args.only.split(",")) if args.only else None

    rows, skipped = [], []
    for name, run in TASKS:
        if wanted and name not in wanted:
            continue
        try:
            infer, e2e, mb, result = run()
        except Exception as exc:  # a missing model raises through require_model's skip
            skipped.append((name, str(exc).splitlines()[0][:80]))
            print(f"  skip {name}: {skipped[-1][1]}")
            continue
        rows.append(dict(task=name, infer_ms=round(infer, 2) if infer == infer else None,
                         e2e_ms=round(e2e, 2), model_mb=round(mb, 2), result=result))
        print(f"  {name:16s} infer {infer:8.2f} ms   e2e {e2e:8.2f} ms   {result}")

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
