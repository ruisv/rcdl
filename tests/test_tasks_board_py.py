"""End-to-end task heads against REAL models on the board.

Every test here runs a real `.rknn` through the real preprocessing and asserts
on what came out. That is deliberately different from `test_instance_seg.py`,
`test_segmentation.py` and friends, which pin the decode maths against a numpy
oracle and need no hardware: those catch a wrong formula, these catch a wrong
*model contract* — an output layout the resolver mis-read, a channel order that
silently costs accuracy, an activation applied twice or not at all.

Everything skips cleanly when the model, the bindings or OpenCV are missing:

    PYTHONPATH=build:python pytest -s tests/test_tasks_board_py.py
"""

import math

import numpy as np
import pytest

import board_models as bm


@pytest.fixture(scope="module")
def rcdl():
    return pytest.importorskip("rcdl", reason="build the module on the board first")


def need(mod, *names):
    for n in names:
        if not hasattr(mod, n):
            pytest.skip(f"compiled module has no {n} binding yet")


# --------------------------------------------------------------------------- #
# Instance segmentation — yolov8n-seg                                          #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def seg_model():
    return bm.require_model("yolov8n-seg_rk3588.rknn")


def test_instance_seg_finds_the_same_objects_as_detection(rcdl, seg_model):
    """The seg model is the detection head plus a mask branch, so it must find
    the same scene: one bus and four people on `bus.jpg`. If the mask-coefficient
    branch were mistaken for the class branch (both are plausible channel counts)
    the boxes would survive but the classes would be nonsense — this catches that.
    """
    need(rcdl, "InstanceSegmenter")
    img = bm.load_bgr("bus.jpg")
    e = rcdl.Engine(seg_model)
    segmenter = rcdl.InstanceSegmenter(e._e) if hasattr(rcdl, "InstanceSegmenter") else None
    masks = segmenter.process(np.ascontiguousarray(img).reshape(-1),
                              img.shape[1], img.shape[0], "bgr888")
    counts = {}
    for m in masks:
        counts[rcdl.coco_class_name(m.class_id)] = counts.get(rcdl.coco_class_name(m.class_id), 0) + 1
    print(f"\n{len(masks)} instances: {counts}")
    assert counts.get("bus", 0) == 1, counts
    assert counts.get("person", 0) == 4, counts


def test_instance_masks_are_inside_their_boxes_and_not_empty(rcdl, seg_model):
    """A mask that is all zeros means the prototype combination or the sigmoid is
    wrong; one that fills its whole box means the crop never happened. Measured
    on this sample, people cover 30–60% of their box and the bus about 63%."""
    need(rcdl, "InstanceSegmenter")
    img = bm.load_bgr("bus.jpg")
    e = rcdl.Engine(seg_model)
    masks = rcdl.InstanceSegmenter(e._e).process(
        np.ascontiguousarray(img).reshape(-1), img.shape[1], img.shape[0], "bgr888")
    assert masks
    for m in masks:
        area = max(1.0, (m.x2 - m.x1) * (m.y2 - m.y1))
        arr = np.asarray(m.mask() if callable(getattr(m, "mask", None)) else m.mask)
        on = int(arr.astype(bool).sum())
        frac = on / area
        print(f"  {rcdl.coco_class_name(m.class_id):<8} {m.score:.3f} "
              f"mask covers {frac * 100:5.1f}% of its box")
        assert 0.10 < frac < 0.95, f"implausible mask coverage {frac:.2f}"


# --------------------------------------------------------------------------- #
# Semantic segmentation — PP-LiteSeg (Cityscapes)                              #
# --------------------------------------------------------------------------- #
def test_semantic_seg_on_a_street_scene(rcdl):
    """PP-LiteSeg on its own Cityscapes sample. The assertion is structural, not
    a per-pixel match: a street scene must resolve into a handful of large
    regions, not one class everywhere (a dead argmax) and not confetti (a wrong
    channel order over the logits). Measured: 11 classes, road ~41%, vegetation
    ~31%."""
    need(rcdl, "Segmenter")
    model = bm.require_model("ppseg_rk3588.rknn")
    img = bm.load_bgr("cityscapes.png")
    e = rcdl.Engine(model)
    mask = rcdl.Segmenter(e._e).process(np.ascontiguousarray(img).reshape(-1),
                                        img.shape[1], img.shape[0], "bgr888")
    labels = np.asarray(mask.labels if not callable(getattr(mask, "labels", None))
                        else mask.labels())
    ids, counts = np.unique(labels, return_counts=True)
    frac = counts / counts.sum()
    print(f"\n{labels.shape} -> {len(ids)} classes; top: "
          f"{[(int(i), round(float(f), 3)) for i, f in sorted(zip(ids, frac), key=lambda t: -t[1])[:4]]}")
    assert 4 <= len(ids) <= 19, f"{len(ids)} distinct classes is not a street scene"
    assert frac.max() < 0.75, "one class covers the frame — argmax is not discriminating"
    assert sorted(frac)[-2] > 0.05, "only one class has meaningful area"


# --------------------------------------------------------------------------- #
# Classification — ResNet-18                                                   #
# --------------------------------------------------------------------------- #
def test_classifier_on_the_space_shuttle(rcdl):
    """ImageNet class 812 is 'space shuttle'. This is the tightest available
    check that the whole classification path — centre crop, channel order,
    softmax placement — is right, because it pins one exact class index."""
    need(rcdl, "Classifier")
    model = bm.find_model("resnet18_rk3588.rknn")
    if model is None:
        pytest.skip("resnet18_rk3588.rknn not staged")
    img = bm.load_bgr("space_shuttle_224.jpg")
    e = rcdl.Engine(model)
    top = rcdl.Classifier(e._e).classify(np.ascontiguousarray(img).reshape(-1),
                                         img.shape[1], img.shape[0], "bgr888")
    print(f"\ntop-5: {[(t.class_id, round(t.score, 3)) for t in top[:5]]}")
    assert top[0].class_id == 812, f"expected class 812 (space shuttle), got {top[0].class_id}"


# --------------------------------------------------------------------------- #
# Detection parity across models                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["yolov8n_rk3588.rknn", "yolo11n_rk3588.rknn"])
def test_both_detectors_agree_on_the_scene(rcdl, name):
    """YOLOv8n and YOLO11n are different networks with the same head layout, so
    they must independently agree on what is in `bus.jpg`. Running both is what
    distinguishes 'the decoder is right' from 'the decoder happens to suit one
    export'."""
    model = bm.find_model(name)
    if model is None:
        pytest.skip(f"{name} not staged")
    img = bm.load_bgr("bus.jpg")
    dets = rcdl.detect(rcdl.Engine(model).detector(model_input="rgb888"), img, "bgr888")
    counts = {}
    for d in dets:
        n = rcdl.coco_class_name(d.class_id)
        counts[n] = counts.get(n, 0) + 1
    print(f"\n{name}: {counts}")
    assert counts.get("bus", 0) == 1, counts
    assert counts.get("person", 0) >= 3, counts


# --------------------------------------------------------------------------- #
# Pose — yolov8n-pose                                                          #
# --------------------------------------------------------------------------- #
def test_pose_finds_two_people_with_joints_inside_their_own_boxes(rcdl):
    """The pose head packs box, class and 17x3 keypoint channels into one fused
    tensor, and reading that layout wrong is the classic silent failure here: the
    boxes stay plausible while the joints scatter. `test_pose.py` pins the decode
    maths against numpy; this pins the *layout*, which numpy cannot see.

    The assertion that actually bites is per-person containment — every keypoint
    the model is confident about must land inside that person's own box. A
    swapped stride or a transposed (kpt, coord) pair keeps the joints on the
    image but scatters them across both people, and this catches it.

    Measured on `zidane.jpg`: two people at 0.904 / 0.889, 17 joints each, 8 and
    11 of them above 0.5, and every one of those inside its own box. Two
    low-confidence joints (0.41, 0.11) sit a few pixels below the left person's
    box — hence the confidence filter rather than a flat 17/17.
    """
    need(rcdl, "PoseEstimator")
    model = bm.require_model("yolov8n-pose_rk3588.rknn")
    img = bm.load_bgr("zidane.jpg")
    h, w = img.shape[:2]
    people = rcdl.estimate_pose(rcdl.Engine(model).pose_estimator(), img, "bgr888")
    print(f"\n{len(people)} people: {[round(p.box.score, 3) for p in people]}")

    assert len(people) == 2, f"expected 2 people, got {len(people)}"
    for p in people:
        b = p.box
        assert b.score > 0.8, f"weak person {b.score:.3f}"
        assert len(p.keypoints) == 17, f"{len(p.keypoints)} keypoints, COCO pose has 17"
        assert b.x2 > b.x1 and b.y2 > b.y1, "degenerate person box"

        confident = [k for k in p.keypoints if k.score > 0.5]
        assert len(confident) >= 6, f"only {len(confident)} joints above 0.5"
        outside = [(i, k) for i, k in enumerate(p.keypoints)
                   if k.score > 0.5 and not (b.x1 <= k.x <= b.x2 and b.y1 <= k.y <= b.y2)]
        print(f"  person {b.score:.3f} box=({b.x1:.0f},{b.y1:.0f},{b.x2:.0f},{b.y2:.0f}) "
              f"{len(confident)}/17 joints >0.5, {len(outside)} of them outside the box")
        assert not outside, f"confident joints outside their own box: {outside}"

        # Nothing may leave the frame: the un-letterbox is applied to keypoints
        # as well as boxes, and forgetting it there sends joints off-canvas.
        for i, k in enumerate(p.keypoints):
            assert 0.0 <= k.x <= w and 0.0 <= k.y <= h, f"keypoint {i} off-frame: ({k.x}, {k.y})"
            assert 0.0 <= k.score <= 1.0, f"keypoint {i} visibility {k.score} not a probability"


def test_pose_boxes_agree_with_the_plain_detector(rcdl):
    """The pose model's box branch is the detection head, so it must put people
    where YOLOv8n does. Cross-checking two independent models is what separates
    'the pose decoder is right' from 'the pose decoder is self-consistent'."""
    need(rcdl, "PoseEstimator")
    pose_model = bm.require_model("yolov8n-pose_rk3588.rknn")
    det_model = bm.find_model("yolov8n_rk3588.rknn")
    if det_model is None:
        pytest.skip("yolov8n_rk3588.rknn not staged")
    img = bm.load_bgr("zidane.jpg")

    people = rcdl.estimate_pose(rcdl.Engine(pose_model).pose_estimator(), img, "bgr888")
    dets = rcdl.detect(rcdl.Engine(det_model).detector(model_input="rgb888"), img, "bgr888")
    persons = [d for d in dets if rcdl.coco_class_name(d.class_id) == "person"]
    print(f"\npose {len(people)} vs detector {len(persons)} people")
    assert len(persons) == len(people), f"{len(persons)} detected vs {len(people)} posed"

    # Match by centre; the two heads are different networks, so allow real slack.
    for p in sorted(people, key=lambda p: p.box.x1):
        pc = ((p.box.x1 + p.box.x2) / 2, (p.box.y1 + p.box.y2) / 2)
        best = min(persons, key=lambda d: abs((d.x1 + d.x2) / 2 - pc[0])
                   + abs((d.y1 + d.y2) / 2 - pc[1]))
        dx = abs((best.x1 + best.x2) / 2 - pc[0])
        dy = abs((best.y1 + best.y2) / 2 - pc[1])
        print(f"  centre delta ({dx:.1f}, {dy:.1f}) px")
        assert dx < 40 and dy < 40, f"pose box centre {pc} matches no detection"


# --------------------------------------------------------------------------- #
# Oriented boxes — yolov8n-obb (DOTA)                                          #
# --------------------------------------------------------------------------- #
def test_obb_reads_an_aerial_scene_with_consistent_orientations(rcdl):
    """DOTA aerial imagery, decoded through the rotated head.

    Two things can go wrong silently. The angle channel sits at the end of the
    fused tensor, so an off-by-one in the layout yields boxes that are right in
    position and nonsense in rotation. And rotated NMS is its own algorithm —
    the float32 version of it once computed an intersection area of -39.6 — so a
    broken IoU shows up as duplicate boxes, not as an error.

    The vehicles in this scene are parked in rows, which is what makes the
    orientation checkable: their angles must cluster. Measured: 4 planes,
    26 large vehicles (angle std 0.20 rad), 3 ships. The counts are asserted as
    ranges because several boxes sit right at the confidence threshold.

    `obb.jpg` is a progressive JPEG, which the VPU cannot decode — it is read
    with OpenCV on purpose.
    """
    need(rcdl, "ObbDetector")
    model = bm.require_model("yolov8n-obb_rk3588.rknn")
    img = bm.load_bgr("obb.jpg")
    h, w = img.shape[:2]
    dets = rcdl.detect_obb(rcdl.Engine(model).obb_detector(), img, "bgr888")

    counts = {}
    for d in dets:
        n = rcdl.dota_class_name(d.class_id)
        counts[n] = counts.get(n, 0) + 1
    print(f"\n{img.shape[1]}x{img.shape[0]} -> {len(dets)} oriented boxes: {counts}")

    assert 25 <= len(dets) <= 45, f"{len(dets)} boxes is not this scene"
    assert 3 <= counts.get("plane", 0) <= 6, counts
    assert 18 <= counts.get("large vehicle", 0) <= 34, counts
    assert 1 <= counts.get("ship", 0) <= 6, counts

    for d in dets:
        r = d.rrect
        assert r.w > 0 and r.h > 0, f"degenerate rotated box {r.w}x{r.h}"
        assert 0.0 <= d.score <= 1.0
        # A box may overhang the frame edge, but its centre cannot be off-image:
        # that is what a missing un-letterbox on the rotated centre looks like.
        assert -1 <= r.cx <= w + 1 and -1 <= r.cy <= h + 1, f"centre off-frame ({r.cx}, {r.cy})"
        assert max(r.w, r.h) < max(w, h), "box larger than the frame"

    vehicles = [d.rrect.angle for d in dets if rcdl.dota_class_name(d.class_id) == "large vehicle"]
    import math
    # Orientation is defined modulo pi for a box, so cluster on the doubled angle
    # (a vector average) instead of on the raw radians, where 0 and pi are far
    # apart numerically but identical geometrically.
    cs = sum(math.cos(2 * a) for a in vehicles) / len(vehicles)
    sn = sum(math.sin(2 * a) for a in vehicles) / len(vehicles)
    concentration = math.hypot(cs, sn)
    print(f"  {len(vehicles)} large vehicles, orientation concentration {concentration:.3f} "
          f"(1.0 = perfectly aligned)")
    assert concentration > 0.7, (
        f"parked vehicles should share an orientation; concentration {concentration:.2f} "
        "suggests the angle channel is being read from the wrong offset")


def test_obb_boxes_do_not_duplicate_each_other(rcdl):
    """Rotated NMS, checked from the outside: no two surviving boxes of the same
    class may sit on top of each other. This is the assertion that would have
    caught the negative-area rotated IoU, which suppressed nothing."""
    need(rcdl, "ObbDetector")
    model = bm.require_model("yolov8n-obb_rk3588.rknn")
    img = bm.load_bgr("obb.jpg")
    dets = rcdl.detect_obb(rcdl.Engine(model).obb_detector(), img, "bgr888")

    import math
    worst = 0.0
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            a, b = dets[i], dets[j]
            if a.class_id != b.class_id:
                continue
            d = math.hypot(a.rrect.cx - b.rrect.cx, a.rrect.cy - b.rrect.cy)
            # Two boxes whose centres are closer than a quarter of the smaller
            # box's short side are the same object surviving twice.
            near = 0.25 * min(min(a.rrect.w, a.rrect.h), min(b.rrect.w, b.rrect.h))
            worst = max(worst, near - d)
            assert d > near, (
                f"two {rcdl.dota_class_name(a.class_id)} boxes {d:.1f}px apart "
                f"(min {near:.1f}) — rotated NMS did not suppress a duplicate")
    print(f"\nclosest same-class pair cleared the duplicate bound by {-worst:.1f} px")


# --------------------------------------------------------------------------- #
# Faces — RetinaFace                                                           #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def face_model():
    return bm.require_model("retinaface_rk3588.rknn")


def test_retinaface_finds_both_faces_with_ordered_landmarks(rcdl, face_model):
    """Two faces on `zidane.jpg`, each with five landmarks.

    The landmark order — left eye, right eye, nose, left mouth, right mouth — is
    checked geometrically rather than by coordinate: both eyes above the nose,
    the nose above both mouth corners. A landmark decoder that reads the ten
    channels as (x0..x4, y0..y4) instead of interleaved pairs still produces
    points inside the face box, so only the arrangement gives it away.

    Measured: 0.995 at (918,103,1050,264) and 0.949 at (552,262,655,398).
    """
    need(rcdl, "FaceDetector")
    img = bm.load_bgr("zidane.jpg")
    faces = rcdl.detect_faces(rcdl.Engine(face_model).face_detector(), img, "bgr888")
    print(f"\n{len(faces)} faces: {[round(f.score, 3) for f in faces]}")
    assert len(faces) == 2, f"expected 2 faces, got {len(faces)}"

    for f in faces:
        assert f.score > 0.9, f"weak face {f.score:.3f}"
        assert f.x2 > f.x1 and f.y2 > f.y1
        lm = [tuple(p) for p in f.landmarks]
        assert len(lm) == 5, f"{len(lm)} landmarks, RetinaFace has 5"
        for (x, y) in lm:
            assert f.x1 <= x <= f.x2 and f.y1 <= y <= f.y2, \
                f"landmark ({x:.1f}, {y:.1f}) outside face ({f.x1:.0f},{f.y1:.0f},{f.x2:.0f},{f.y2:.0f})"
        (lex, ley), (rex, rey), (nx, ny), (lmx, lmy), (rmx, rmy) = lm
        print(f"  face {f.score:.3f} eyes y={ley:.0f}/{rey:.0f} nose y={ny:.0f} "
              f"mouth y={lmy:.0f}/{rmy:.0f}")
        assert ley < ny and rey < ny, "eyes must sit above the nose"
        assert ny < lmy and ny < rmy, "nose must sit above the mouth corners"
        assert lex < rex, "landmark 0 should be the subject's left eye in image order"


def test_retinaface_needs_bgr_and_says_so_by_losing_a_face(rcdl, face_model):
    """RetinaFace is trained with BGR channel order and BGR mean subtraction, and
    feeding it RGB does not fail — it silently halves recall. The reference demo
    converts BGR to RGB, which makes this the easy mistake to inherit.

    This test pins the trap itself: with `model_input="bgr888"` (the default) both
    faces come back at 0.995 and 0.949; with `model_input="rgb888"` the second
    face collapses to 0.129 and only one survives. Nothing raises, the surviving
    box is still correct, and the landmarks still look right — which is exactly
    why it needs a test rather than a comment.
    """
    need(rcdl, "FaceDetector")
    img = bm.load_bgr("zidane.jpg")
    e = rcdl.Engine(face_model)

    bgr = rcdl.detect_faces(e.face_detector(model_input="bgr888"), img, "bgr888")
    rgb = rcdl.detect_faces(e.face_detector(model_input="rgb888"), img, "bgr888")
    print(f"\nmodel fed BGR: {[round(f.score, 3) for f in bgr]}")
    print(f"model fed RGB: {[round(f.score, 3) for f in rgb]}")

    assert len(bgr) == 2, f"the BGR path is the correct one and must find 2 faces, got {len(bgr)}"
    assert len(rgb) < len(bgr), (
        "feeding RetinaFace RGB used to cost a face silently; if this now matches "
        "the BGR path the preprocessing changed and the default should be revisited")


# --------------------------------------------------------------------------- #
# Text — PP-OCRv4 detection + recognition                                      #
# --------------------------------------------------------------------------- #
def _warp_upright(img, box):
    """Crop one TextBox quadrilateral out of `img` and warp it upright.

    This step lives in the test rather than the library on purpose: it is a host
    image operation, not a decode. `TextBox.pts` is TL, TR, BR, BL — the order a
    4-point perspective transform expects.
    """
    import cv2
    import numpy as np
    pts = np.asarray(box.pts, dtype=np.float32).reshape(4, 2)
    w = int(round(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[3] - pts[2]))))
    h = int(round(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))))
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (max(w, 1), max(h, 1)))


def test_ocr_detects_every_text_line_on_the_sample(rcdl):
    """PP-OCRv4 DBNet on its own sample. The head is a single probability map,
    so what is really under test is the binarise → contour → unclip chain:
    too small an unclip ratio clips the glyphs and the recogniser then drops
    characters, while a wrong sigmoid placement merges every line into one blob.

    Measured on `ocr.jpg` (500x500): 16 boxes, all upright, none degenerate.
    """
    need(rcdl, "TextDetector")
    model = bm.require_model("ppocrv4_det_rk3588.rknn")
    img = bm.load_bgr("ocr.jpg")
    h, w = img.shape[:2]
    boxes = rcdl.detect_text(rcdl.Engine(model).text_detector(), img, "bgr888")
    print(f"\n{w}x{h} -> {len(boxes)} text boxes")

    assert 14 <= len(boxes) <= 18, f"{len(boxes)} boxes; the sample has 16 text lines"
    for b in boxes:
        assert b.x2 > b.x1 and b.y2 > b.y1, "degenerate text box"
        assert 0.0 <= b.score <= 1.0
        assert -2 <= b.x1 and b.x2 <= w + 2 and -2 <= b.y1 and b.y2 <= h + 2, \
            f"box ({b.x1:.0f},{b.y1:.0f},{b.x2:.0f},{b.y2:.0f}) outside the {w}x{h} frame"
        assert len(list(b.pts)) == 8, "a TextBox is a quadrilateral: 4 interleaved xy pairs"

    # A merged-blob failure shows up as one box covering the page.
    biggest = max((b.x2 - b.x1) * (b.y2 - b.y1) for b in boxes)
    assert biggest < 0.5 * w * h, "one box covers half the page — the lines merged"


def test_ocr_recognizes_the_sample_text(rcdl):
    """Detection feeding recognition, the way it is actually deployed.

    CTC decoding is the fragile half: an off-by-one in the character dictionary
    shifts every glyph, and a `time_major` mistake transposes the whole
    sequence. Both produce confident-looking Chinese that is simply wrong, so
    the assertion has to be on exact strings.

    Measured: 15 of the 16 boxes carry a text line and all 15 read correctly; the
    16th is a narrow vertical strip (17x73) that decodes to the empty string.
    """
    need(rcdl, "TextDetector", "TextRecognizer")
    import os
    det_model = bm.require_model("ppocrv4_det_rk3588.rknn")
    rec_model = bm.require_model("ppocrv4_rec_rk3588.rknn")
    dict_path = os.path.join(os.path.dirname(bm.IMAGES), "ppocr_keys_v1_6625.txt")
    if not os.path.isfile(dict_path):
        pytest.skip(f"character dictionary missing: {dict_path}")

    img = bm.load_bgr("ocr.jpg")
    boxes = rcdl.detect_text(rcdl.Engine(det_model).text_detector(), img, "bgr888")
    recognizer = rcdl.Engine(rec_model).text_recognizer(dict_path)

    texts = []
    for b in sorted(boxes, key=lambda b: (b.y1, b.x1)):
        crop = _warp_upright(img, b)
        if crop.shape[0] < 4 or crop.shape[1] < 4:
            continue
        line = rcdl.recognize_text(recognizer, crop, "bgr888")
        print(f"  {crop.shape[1]:>3}x{crop.shape[0]:<3} {line.score:.3f} {line.text!r}")
        if line.text:
            texts.append(line.text)

    print(f"\n{len(texts)} lines recognized out of {len(boxes)} boxes")
    assert len(texts) >= 14, f"only {len(texts)} lines produced text"

    # Exact strings: these pin the dictionary offset and the CTC time axis. A
    # dictionary off by one still yields fluent-looking Chinese, so a substring
    # check would not be enough.
    joined = "".join(texts)
    for expected in ["纯臻营养护发素",
                     "产品信息/参数",
                     "【品名】：纯臻营养护发素",
                     "【净含量】：220ml",
                     "【适用人群】：适合所有肤质"]:
        assert expected in joined, f"missing exact line {expected!r}"

    # Digits and Latin come out of a different part of the 6625-entry table than
    # the Chinese does, so check one of each survived.
    assert "YM-X-3011" in joined, "the Latin/digit part of the dictionary is off"


def test_face_alignment_puts_every_face_in_the_same_pose(rcdl):
    """Warp each detected face onto the ArcFace template and check where it lands.

    Identity embeddings are computed on an ALIGNED crop, never on the detector's
    box: the model is trained with the eyes on fixed pixels, and handing it a
    plain crop still returns a unit-length 512-d vector that is simply worse at
    telling people apart. So the alignment has to be checked, and it can be —
    against the detector itself. Warp, re-detect inside the 112x112 result, and
    see whether the landmarks arrive where the template says.

    Measured on the four faces this project's images contain (two in `zidane.jpg`
    at 1.00/0.95, two in `bus.jpg` at 0.85/0.70):

      * the transform's own residual is 1.3-7.3 px mean — five points and four
        degrees of freedom, so a turned head cannot fit exactly, and the largest
        residual is indeed the turned face;
      * RetinaFace re-finds every aligned crop at 0.99-1.00;
      * the re-detected eyes land at y 52-62 and mouths at y 94-97, against the
        template's 51.6 and 92.3. That band is the assertion: it is loose enough
        for the detector's own error on a 112x112 upscale of a 36x49 face, and
        tight enough that a mirrored, rotated or transposed transform fails it.
    """
    need(rcdl, "FaceDetector", "face_align_transform")
    import cv2
    model = bm.require_model("retinaface_rk3588.rknn")
    det = rcdl.Engine(model).face_detector(model_input="bgr888")
    tpl = np.asarray(rcdl.arcface_template(112, 112))

    checked = 0
    for name in ("zidane.jpg", "bus.jpg"):
        img = bm.load_bgr(name)
        for face in [f for f in rcdl.detect_faces(det, img, "bgr888") if f.score > 0.5]:
            lm = np.asarray(face.landmarks, dtype=np.float32)
            assert lm.shape == (5, 2)
            m = np.asarray(rcdl.face_align_transform(lm.reshape(-1), 112, 112))
            assert m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0] > 0, "the alignment mirrored the face"

            residual = np.linalg.norm(lm @ m[:, :2].T + m[:, 2] - tpl, axis=1)
            assert residual.max() < 15.0, (
                f"{name}: the transform does not fit its own landmarks (max {residual.max():.1f} px)")

            crop = cv2.warpAffine(img, m, (112, 112))
            found = [g for g in rcdl.detect_faces(det, crop, "bgr888") if g.score > 0.3]
            assert found, f"{name}: no face left in the aligned crop"
            g = max(found, key=lambda g: g.score)
            assert g.score > 0.9, f"{name}: the aligned crop only scores {g.score:.2f}"

            got = np.asarray(g.landmarks, dtype=np.float32)
            for eye in (got[0], got[1]):
                assert abs(eye[1] - tpl[0][1]) < 15, f"{name}: an eye landed at y={eye[1]:.1f}"
            for mouth in (got[3], got[4]):
                assert abs(mouth[1] - tpl[3][1]) < 15, f"{name}: a mouth corner at y={mouth[1]:.1f}"
            assert got[:2, 1].mean() < got[3:, 1].mean() - 20, (
                f"{name}: the eyes are not above the mouth — the crop is upside down")
            checked += 1
    assert checked == 4, f"expected the four faces these images contain, aligned {checked}"


def _ocr_line_crops(rcdl, img):
    """Every text line of the sample page, cropped and warped upright."""
    det = rcdl.Engine(bm.require_model("ppocrv4_det_rk3588.rknn")).text_detector()
    boxes = rcdl.detect_text(det, img, "bgr888")
    crops = []
    for b in sorted(boxes, key=lambda b: (b.y1, b.x1)):
        c = _warp_upright(img, b)
        if c.shape[0] >= 6 and c.shape[1] >= 12:
            crops.append(c)
    return crops


def test_yolo26_obb_regresses_radians_where_v8_regressed_a_fraction(rcdl):
    """The angle convention changed generations, and only the number moves.

    YOLOv8's OBB head emits a value mapped by `(sigmoid(v) - 0.25) * pi`; YOLO26
    regresses the angle in RADIANS and applies nothing. Decoded with the wrong
    one the boxes are still boxes, still on the objects, just rotated wrongly —
    and rotated NMS then merges a different set of them, so even the COUNT
    changes. Measured against the PyTorch reference for the same image, the
    largest plane sits at 0.533 rad; RCDL reads 0.53 with the YOLO26 convention
    and 0.88 with v8's.
    """
    need(rcdl, "ObbDetector")
    path = bm.find_model("yolo26n-obb_rk3588.rknn")
    if path is None:
        pytest.skip("yolo26n-obb is not staged")
    img = bm.load_bgr("obb.jpg")
    engine = rcdl.Engine(path)

    # The framework's own answer for this image: four aircraft, at these centres
    # and these angles in radians.
    reference = {(785, 1894): 0.533, (1200, 1284): 0.438,
                 (2132, 1810): 0.537, (2706, 2066): 0.519}

    def planes(**kw):
        det = engine.obb_detector(**kw)
        out = det.process(np.ascontiguousarray(img), img.shape[1], img.shape[0], "bgr888")
        found = [d for d in out if rcdl.dota_class_name(d.class_id) == "plane"]
        assert len(found) == 4, f"expected the four planes, got {len(found)} ({kw})"
        return out, found

    def same_rectangle(a, b):
        """Angles of the SAME rectangle, allowing for regularisation.

        RCDL canonicalises every box to w >= h, which swaps the sides and adds
        pi/2 — the identical physical rectangle described the other way round.
        So two angles describe one rectangle when they agree modulo pi/2, and
        comparing them any other way would fail on half the aircraft here purely
        because the reference did not regularise.
        """
        d = abs(a - b) % (math.pi / 2)
        return min(d, math.pi / 2 - d)

    def matched(found):
        out = {}
        for cx, cy in reference:
            near = min(found, key=lambda d: (d.rrect.cx - cx) ** 2 + (d.rrect.cy - cy) ** 2)
            assert (near.rrect.cx - cx) ** 2 + (near.rrect.cy - cy) ** 2 < 40 ** 2, (
                f"no plane near ({cx}, {cy})")
            out[(cx, cy)] = near.rrect.angle
        return out

    boxes, found = planes(angle_bias=0.0, angle_scale=1.0)
    names = [rcdl.dota_class_name(d.class_id) for d in boxes]
    assert names.count("large vehicle") >= 20, f"only {names.count('large vehicle')} vehicles"
    right = matched(found)
    for key, expect in reference.items():
        assert same_rectangle(right[key], expect) < 0.12, (
            f"plane at {key}: {right[key]:.3f} rad against the framework's {expect:.3f}")

    _, wrong_found = planes()  # the v8 convention, on a YOLO26 model
    wrong = matched(wrong_found)
    off = [same_rectangle(wrong[k], reference[k]) for k in reference]
    assert max(off) > 0.2, (
        "the v8 angle convention agrees with the framework on every plane, so the two "
        "conventions have stopped differing and this test is vacuous")


def test_yolo26_classifier_has_its_softmax_in_the_graph(rcdl):
    """Right ranking, meaningless scores — the double-softmax signature.

    ResNet-18 emits logits and RCDL softmaxes them; the YOLO26 classifier emits
    probabilities already. Softmax them again and the ARGMAX survives — the top
    three are still space shuttle, submarine, catamaran — while the confidence
    collapses towards uniform: 0.003 instead of 0.939 over 1000 classes. Nothing
    errors, and a test that only checked the predicted class would pass.
    """
    need(rcdl, "Classifier")
    path = bm.find_model("yolo26n-cls_rk3588.rknn")
    if path is None:
        pytest.skip("yolo26n-cls is not staged")
    img = bm.load_bgr("space_shuttle_224.jpg")
    engine = rcdl.Engine(path)

    right = engine.classifier(top_k=3, crop_ratio=1.0, apply_softmax=False).classify(
        np.ascontiguousarray(img), img.shape[1], img.shape[0], "bgr888")
    assert right[0].class_id == 812, f"top-1 is {right[0].class_id}, expected 812 (space shuttle)"
    assert right[0].score > 0.5, f"top-1 scored only {right[0].score:.3f}"

    doubled = engine.classifier(top_k=3, crop_ratio=1.0).classify(
        np.ascontiguousarray(img), img.shape[1], img.shape[0], "bgr888")
    assert doubled[0].class_id == 812, "the argmax should survive a second softmax"
    assert doubled[0].score < 0.05, (
        f"a second softmax should flatten the score, but it is {doubled[0].score:.3f}")


def test_yolo26_pose_needs_its_own_keypoint_formula(rcdl):
    """The same model, two decode formulas, and only one puts the joints on the body.

    YOLOv8 and YOLO11 predict a keypoint offset in HALF cells — the decode is
    `(2*raw + grid) * stride` — and YOLO26 dropped the doubling: `(raw + grid +
    0.5) * stride`. One factor of two, and nothing in the tensor says which. Get
    it wrong and the joints land at roughly twice their distance from the cell
    centre, which on a person a few hundred pixels tall is still a skeleton, is
    still drawable, and is simply not this person's.

    So the test runs BOTH formulas over one model and asserts the difference:
    with `cell_relative_whole` every confident joint sits inside its own person's
    box (16 of 16 on the two clear people), and with the older `cell_relative`
    only 5 of those 16 do.
    """
    need(rcdl, "PoseEstimator")
    path = bm.find_model("yolo26n-pose_rk3588.rknn")
    if path is None:
        pytest.skip("yolo26n-pose is not staged")
    img = bm.load_bgr("bus.jpg")
    engine = rcdl.Engine(path)

    def inside_own_box(kpt_decode):
        est = engine.pose_estimator(kpt_decode=kpt_decode, kpt_apply_sigmoid=True)
        poses = est.process(np.ascontiguousarray(img), img.shape[1], img.shape[0], "bgr888")
        poses = sorted(poses, key=lambda p: -p.box.score)[:2]
        assert len(poses) == 2, f"expected the two clear people, got {len(poses)}"
        confident = inside = 0
        for p in poses:
            b = p.box
            for k in p.keypoints:
                if k.score > 0.5:
                    confident += 1
                    inside += b.x1 <= k.x <= b.x2 and b.y1 <= k.y <= b.y2
        return confident, inside

    conf_right, inside_right = inside_own_box("cell_relative_whole")
    conf_wrong, inside_wrong = inside_own_box("cell_relative")
    assert conf_right >= 12, f"only {conf_right} confident joints on two people"
    assert inside_right == conf_right, (
        f"YOLO26 formula: {inside_right}/{conf_right} joints inside their own box")
    assert inside_wrong < conf_wrong / 2, (
        f"the v8 formula put {inside_wrong}/{conf_wrong} joints inside the box — if that is "
        "no longer wrong, the two decodes have stopped differing and this test is vacuous")


@pytest.mark.parametrize("model_name,min_people", [("yolov8n-seg_rk3588.rknn", 4),
                                                   ("yolo26n-seg_rk3588.rknn", 3)])
def test_instance_seg_generations_segment_the_same_bus(rcdl, model_name, min_people):
    """Both segmentation generations, through one decoder that reads its own layout.

    YOLO26's box branch has 4 channels where YOLOv8's has 64 (no DFL), and its
    prototypes are built from all three feature scales rather than P3 alone — yet
    the head resolver handles both from the model's signature, which is what this
    checks. The instance counts differ on purpose: measured here, v8n-seg finds
    the bus and all four people including the one cropped by the left edge (which
    it scores 0.318), while 26n-seg finds the bus and the three unambiguous people
    at 0.84-0.91 and leaves the edge case below threshold. So the assertion is the
    floor each one actually clears, not a number that flatters both.
    """
    need(rcdl, "InstanceSegmenter")
    path = bm.find_model(model_name)
    if path is None:
        pytest.skip(f"{model_name} is not staged")
    img = bm.load_bgr("bus.jpg")
    seg = rcdl.Engine(path).instance_segmenter()
    inst = seg.process(np.ascontiguousarray(img), img.shape[1], img.shape[0], "bgr888")

    classes = [rcdl.coco_class_name(m.class_id) for m in inst]
    assert classes.count("bus") == 1, f"{model_name}: expected one bus, got {classes}"
    assert classes.count("person") >= min_people, f"{model_name}: got {classes}"

    # A mask that is empty, or that fills its whole box, means the prototype
    # combination collapsed — both look fine in a count.
    for m in inst:
        area = max(1.0, (m.x2 - m.x1) * (m.y2 - m.y1))
        frac = float(np.asarray(m.mask).sum()) / area
        assert 0.15 < frac < 0.95, (
            f"{model_name}: {rcdl.coco_class_name(m.class_id)} mask covers {frac:.0%} of its box")


def test_the_v5_detector_agrees_with_the_v4_one(rcdl):
    """Two detector generations, one page, the same answer.

    A newer model is only an upgrade if it is at least as good, and on the one
    text page this project carries the two are indistinguishable: 16 regions
    each, 15 of which carry text, and the SAME 15 strings out of the (v4)
    recogniser at the same mean confidence. That is the honest result — not a
    demonstrated improvement, but a demonstrated non-regression, which is what
    justifies keeping the newer build in the registry. The test is the parity
    itself: if a future toolkit or calibration change moves one of them, this
    fails rather than the difference going unnoticed.
    """
    need(rcdl, "TextDetector", "TextRecognizer")
    import os
    v5 = bm.find_model("ppocrv5_det_rk3588.rknn")
    if v5 is None:
        pytest.skip("PP-OCRv5 detection model not staged")
    rec_model = bm.require_model("ppocrv4_rec_rk3588.rknn")
    dict_path = os.path.join(os.path.dirname(bm.IMAGES), "ppocr_keys_v1_6625.txt")
    if not os.path.isfile(dict_path):
        pytest.skip(f"character dictionary missing: {dict_path}")

    img = bm.load_bgr("ocr.jpg")
    rec = rcdl.Engine(rec_model).text_recognizer(dict_path)

    def read(det_model):
        det = rcdl.Engine(det_model).text_detector()
        boxes = sorted(rcdl.detect_text(det, img, "bgr888"), key=lambda b: (b.y1, b.x1))
        out = []
        for b in boxes:
            c = _warp_upright(img, b)
            if c.shape[0] < 6 or c.shape[1] < 12:
                continue
            line = rcdl.recognize_text(rec, c, "bgr888")
            if line.text:
                out.append(line.text)
        return len(boxes), out

    n4, t4 = read(bm.require_model("ppocrv4_det_rk3588.rknn"))
    n5, t5 = read(v5)
    assert n5 == n4, f"v5 found {n5} regions, v4 found {n4}"
    assert t5 == t4, f"the two detectors led to different text:\n  v4 {t4}\n  v5 {t5}"
    assert len(t5) >= 14, f"only {len(t5)} lines came back — the page should give 15"


def test_text_angle_classifier_reads_the_direction_of_every_line(rcdl):
    """Both orientations of all 16 sample lines, called right every time.

    Measured on the board: 16/16 upright lines labelled 0 and 16/16 of the same
    lines rotated 180 degrees labelled 1, at confidence 1.000 for every line but
    one. That one is the 17x73 vertical strip — a column of stacked glyphs, not a
    text line, which the model scores 0.57/0.61 either way and the 0.9 flip gate
    therefore leaves alone. That is the gate doing its job, so it is asserted
    rather than excluded: an unreadable crop must produce NO flip, because
    flipping an upright line is the one outcome worse than not classifying it.
    """
    need(rcdl, "TextAngleClassifier", "TextDetector")
    import cv2
    cls_model = bm.require_model("ppocr_cls_rk3588.rknn")
    img = bm.load_bgr("ocr.jpg")
    crops = _ocr_line_crops(rcdl, img)
    assert len(crops) >= 15, f"the detector only found {len(crops)} usable lines"

    cls = rcdl.Engine(cls_model).text_angle_classifier()
    assert (cls.input_width, cls.input_height) == (192, 48)

    upright, rotated, flips, unsure = 0, 0, 0, 0
    for c in crops:
        u = cls.process(np.ascontiguousarray(c), c.shape[1], c.shape[0], "bgr888")
        d = cls.process(np.ascontiguousarray(cv2.rotate(c, cv2.ROTATE_180)),
                        c.shape[1], c.shape[0], "bgr888")
        upright += u.label == 0
        rotated += d.label == 1
        flips += d.flip180
        unsure += max(u.score, d.score) < 0.9
        # Never flip an upright line: that is the failure this head must not have.
        assert not u.flip180, f"an upright crop was marked for flipping: {u}"
    assert upright == len(crops), f"{upright}/{len(crops)} upright lines called upright"
    assert rotated == len(crops), f"{rotated}/{len(crops)} rotated lines called rotated"
    assert flips >= len(crops) - 1, f"the flip gate only fired {flips}/{len(crops)} times"
    assert unsure <= 1, f"{unsure} lines were below the gate; only the vertical strip should be"


def test_the_fit_is_pp_ocrs_own_and_a_centred_letterbox_is_not(rcdl):
    """A wide line fills the input's width; a tall crop does not, and is padded.

    This pins the preprocessing rather than the model, because the preprocessing
    is what silently costs accuracy here: the same 16 lines through a CENTRED
    letterbox score 9/16 upright and 11/16 rotated instead of 16/16, with mean
    confidence 0.78 instead of 0.98 (docs/MODELS.md). Nothing errors — the head
    just gets worse at its one job.
    """
    need(rcdl, "TextAngleClassifier")
    cls_model = bm.require_model("ppocr_cls_rk3588.rknn")
    img = bm.load_bgr("ocr.jpg")
    crops = _ocr_line_crops(rcdl, img)
    cls = rcdl.Engine(cls_model).text_angle_classifier()

    wide = max(crops, key=lambda c: c.shape[1] / c.shape[0])
    cls.process(np.ascontiguousarray(wide), wide.shape[1], wide.shape[0], "bgr888")
    assert cls.fit_width == cls.input_width, "a wide line should fill the input's width"

    tall = min(crops, key=lambda c: c.shape[1] / c.shape[0])
    cls.process(np.ascontiguousarray(tall), tall.shape[1], tall.shape[0], "bgr888")
    if tall.shape[1] / tall.shape[0] < cls.input_width / cls.input_height:
        assert cls.fit_width < cls.input_width, (
            f"a {tall.shape[1]}x{tall.shape[0]} crop should be padded, not stretched")


def test_a_rotated_line_is_lost_until_the_classifier_flips_it(rcdl):
    """The reason this head exists, end to end.

    A recogniser handed an upside-down line does not report a problem. Measured
    on the sample page: of the 16 lines rotated 180 degrees, the recogniser
    returns the empty string or a single stray bracket at score <= 0.6 — the
    content is simply gone, with nothing in the result to say so. Run the
    classifier first, rotate what it marks, and every line comes back exactly as
    it reads upright.
    """
    need(rcdl, "TextAngleClassifier", "TextRecognizer", "TextDetector")
    import cv2
    import os
    cls_model = bm.require_model("ppocr_cls_rk3588.rknn")
    rec_model = bm.require_model("ppocrv4_rec_rk3588.rknn")
    dict_path = os.path.join(os.path.dirname(bm.IMAGES), "ppocr_keys_v1_6625.txt")
    if not os.path.isfile(dict_path):
        pytest.skip(f"character dictionary missing: {dict_path}")

    img = bm.load_bgr("ocr.jpg")
    crops = [c for c in _ocr_line_crops(rcdl, img) if c.shape[1] > 4 * c.shape[0]]
    assert crops, "no wide text lines to test with"
    cls = rcdl.Engine(cls_model).text_angle_classifier()
    rec = rcdl.Engine(rec_model).text_recognizer(dict_path)

    naive_survived, recovered, checked = 0, 0, 0
    for c in crops:
        upright = rcdl.recognize_text(rec, c, "bgr888").text
        if not upright:
            continue
        checked += 1
        flipped = cv2.rotate(c, cv2.ROTATE_180)
        naive = rcdl.recognize_text(rec, flipped, "bgr888").text
        if naive == upright:
            naive_survived += 1
        verdict = cls.process(np.ascontiguousarray(flipped), flipped.shape[1], flipped.shape[0],
                              "bgr888")
        fixed = cv2.rotate(flipped, cv2.ROTATE_180) if verdict.flip180 else flipped
        if rcdl.recognize_text(rec, fixed, "bgr888").text == upright:
            recovered += 1

    assert checked >= 10, f"only {checked} lines produced text to compare"
    assert naive_survived == 0, (
        f"{naive_survived} rotated lines read the same as upright — then this test proves nothing")
    assert recovered == checked, (
        f"only {recovered}/{checked} lines came back after the classifier's flip")


# --------------------------------------------------------------------------- #
# Monocular depth — Depth-Anything-V2-Small (ViT-S, DPT head)                   #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def depth_model():
    # The 308x308 build first: it is both faster and closer to the fp32
    # reference than the native 518x518 one (docs/MODELS.md).
    return bm.require_model("depth_anything_v2_vits_308_rk3588.rknn",
                            "depth_anything_v2_vits_rk3588.rknn")


def test_depth_puts_the_ground_near_and_the_sky_far(rcdl, depth_model):
    """The head emits RELATIVE INVERSE depth — disparity, where big means near —
    and that convention is the whole test. Reading it as depth rather than
    disparity inverts the scene without failing, and a normalisation applied
    before the affine instead of after flattens it. Neither raises.

    `bus.jpg` is a street photographed from the pavement: the road at the bottom
    of the frame is metres away, the sky at the top is not. So the bottom band of
    the map must carry substantially more disparity than the top band. Measured
    on the 308 build: top 20% of rows 0.029, bottom 20% 0.214.
    """
    need(rcdl, "DepthEstimator")
    img = bm.load_bgr("bus.jpg")
    e = rcdl.Engine(depth_model)
    dmap = rcdl.estimate_depth(e.depth_estimator(), img, "bgr888")
    arr = np.asarray(dmap.data() if callable(getattr(dmap, "data", None)) else dmap.data)

    print(f"\n{img.shape[1]}x{img.shape[0]} -> depth {dmap.width}x{dmap.height}, "
          f"raw range [{dmap.vmin:.3f}, {dmap.vmax:.3f}]")
    # The map is projected back through the letterbox, so it covers the frame.
    assert (dmap.height, dmap.width) == img.shape[:2], \
        f"depth map {dmap.width}x{dmap.height} does not cover the {img.shape[1]}x{img.shape[0]} frame"
    assert arr.shape == img.shape[:2]

    assert dmap.vmax > dmap.vmin, "flat map — the head produced nothing"
    assert 0.0 <= arr.min() and arr.max() <= 1.0, "normalize=True must land in [0,1]"

    h = arr.shape[0]
    top, bottom = arr[:h // 5].mean(), arr[-h // 5:].mean()
    print(f"  top 20% of rows {top:.3f} vs bottom 20% {bottom:.3f} (disparity: big = near)")
    assert bottom > top * 2.0, (
        f"bottom band {bottom:.3f} is not clearly nearer than the top {top:.3f} — "
        "the map is inverted or flat")


def test_depth_is_smooth_and_not_quantization_noise(rcdl, depth_model):
    """A depth map is a continuous surface, so neighbouring pixels must agree.
    This is the check that catches an int8 output being read with the wrong
    zero-point or scale: the near/far ordering above can survive that, but the
    map turns into speckle. Compared against the noise floor of the map's own
    value range rather than an absolute number."""
    need(rcdl, "DepthEstimator")
    img = bm.load_bgr("bus.jpg")
    dmap = rcdl.estimate_depth(rcdl.Engine(depth_model).depth_estimator(), img, "bgr888")
    arr = np.asarray(dmap.data() if callable(getattr(dmap, "data", None)) else dmap.data)

    step = float(np.abs(np.diff(arr, axis=1)).mean())
    spread = float(arr.std())
    print(f"\nmean neighbour step {step:.4f} vs map std {spread:.4f} "
          f"(ratio {step / max(spread, 1e-9):.4f})")
    assert spread > 0.02, "the map has almost no contrast"
    assert step < 0.1 * spread, "neighbouring pixels disagree — this is speckle, not depth"


def test_depth_tracks_the_scene_and_not_the_letterbox(rcdl, depth_model):
    """Depth of a wide frame and of the same frame cropped to a different aspect
    ratio must agree where they overlap.

    The model input is square, so a 16:9 frame arrives with padding on two sides
    and a 4:3 crop with much less. If the un-letterbox were missing or applied to
    the wrong axis, the two maps would disagree systematically — the padding
    would drag one of them. Cropping the source is the only way to move the
    padding without changing the scene.
    """
    need(rcdl, "DepthEstimator")
    img = bm.load_bgr("zidane.jpg")
    h, w = img.shape[:2]
    est = rcdl.Engine(depth_model).depth_estimator()

    full = rcdl.estimate_depth(est, img, "bgr888")
    fa = np.asarray(full.data() if callable(getattr(full, "data", None)) else full.data)

    # A centred crop: same pixels, different amount of letterbox padding.
    x0, x1 = w // 6, w - w // 6
    crop = np.ascontiguousarray(img[:, x0:x1])
    part = rcdl.estimate_depth(est, crop, "bgr888")
    pa = np.asarray(part.data() if callable(getattr(part, "data", None)) else part.data)
    assert pa.shape == crop.shape[:2]

    # Compare the overlap after re-normalising both to their own range: the
    # min-max normalise is per-frame, so only the *structure* is comparable.
    overlap = fa[:, x0:x1]
    norm = lambda a: (a - a.min()) / max(1e-9, a.max() - a.min())
    a, b = norm(overlap).ravel(), norm(pa).ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    print(f"\nfull-frame vs cropped, over the shared {x1 - x0}x{h} region: "
          f"correlation {corr:.4f}")
    assert corr > 0.9, (
        f"depth of the same pixels changed with the padding (corr {corr:.3f}) — "
        "the letterbox is not being undone correctly")


def test_depth_map_renders(rcdl, depth_model):
    """`depth_to_gray8` / `depth_colorize` are what a caller actually displays,
    so check they produce a real image rather than a constant."""
    need(rcdl, "DepthEstimator")
    img = bm.load_bgr("bus.jpg")
    dmap = rcdl.estimate_depth(rcdl.Engine(depth_model).depth_estimator(), img, "bgr888")

    gray = np.asarray(rcdl.depth_to_gray8(dmap))
    assert gray.shape == img.shape[:2] and gray.dtype == np.uint8
    assert gray.max() - gray.min() > 100, "the grey render has almost no range"

    bgr = np.asarray(rcdl.depth_colorize(dmap))
    assert bgr.shape == (img.shape[0], img.shape[1], 3) and bgr.dtype == np.uint8
    # Turbo is a colour map: a grey-looking result means the palette is not applied.
    spread = int(bgr.max(axis=2).astype(int).mean() - bgr.min(axis=2).astype(int).mean())
    print(f"\ngray range {gray.min()}..{gray.max()}, mean colour channel spread {spread}")
    assert spread > 20, "colourised map is nearly grey — the Turbo curve is not being applied"


# --------------------------------------------------------------------------- #
# Appearance embeddings — OSNet x0.25 (person ReID, MSMT17)                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def reid_model():
    return bm.require_model("osnet_x0_25_msmt17_rk3588.rknn")


def test_reid_separates_people_and_recognises_the_same_one(rcdl, reid_model):
    """The property that makes an appearance vector useful: two crops of the SAME
    person must score far higher against each other than two crops of different
    people. Everything a tracker does with these vectors rests on that gap.

    A numpy oracle can check that L2-normalisation and cosine are implemented
    correctly, which `test_embedding.py` does. It cannot check that the model is
    being fed the way it was trained — and this head has a specific trap: ReID
    towers are trained on crops SQUASHED to 128x256, so letterboxing one instead
    would feed it padding bars it has never seen. That shows up here as the gap
    closing, not as an error.

    Measured on the four people in `bus.jpg` (int8): different people 0.38–0.47,
    the same person re-cropped 4% tighter 0.960, and the same person taken from
    a 2x upscaled copy of the frame 0.993.
    """
    need(rcdl, "ImageEmbedder")
    det_model = bm.find_model("yolov8n_rk3588.rknn")
    if det_model is None:
        pytest.skip("yolov8n_rk3588.rknn not staged")
    import cv2

    img = bm.load_bgr("bus.jpg")
    det = rcdl.Engine(det_model).detector(model_input="rgb888")
    people = [d for d in rcdl.detect(det, img, "bgr888")
              if rcdl.coco_class_name(d.class_id) == "person"]
    assert len(people) >= 3, f"need several people to compare, found {len(people)}"

    e = rcdl.Engine(reid_model)
    embedder = e.embedder()
    assert e.output_shape(0)[-1] == 512, f"OSNet is 512-d, got {e.output_shape(0)}"

    vecs = [np.asarray(rcdl.embed(embedder, img, (d.x1, d.y1, d.x2, d.y2), "bgr888"))
            for d in people]
    for v in vecs:
        assert v.shape == (512,)
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3, "l2_normalize did not run"

    sims = np.array([[float(a @ b) for b in vecs] for a in vecs])
    different = sims[~np.eye(len(vecs), dtype=bool)]
    print(f"\n{len(people)} people; cross-person cosine "
          f"mean {different.mean():.3f} max {different.max():.3f}")

    # Same person, two ways of arriving at the crop.
    d = people[0]
    w, h = d.x2 - d.x1, d.y2 - d.y1
    tighter = np.asarray(rcdl.embed(embedder, img,
                                    (d.x1 + 0.04 * w, d.y1 + 0.04 * h,
                                     d.x2 - 0.04 * w, d.y2 - 0.04 * h), "bgr888"))
    big = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
    upscaled = np.asarray(rcdl.embed(embedder, big,
                                     (d.x1 * 2, d.y1 * 2, d.x2 * 2, d.y2 * 2), "bgr888"))
    same_tight = float(tighter @ vecs[0])
    same_scale = float(upscaled @ vecs[0])
    print(f"same person: 4% tighter crop {same_tight:.3f}, 2x upscaled frame {same_scale:.3f}")

    assert same_tight > 0.85, f"a 4% crop jitter moved the vector to {same_tight:.3f}"
    assert same_scale > 0.90, f"the same person at 2x scale embedded to {same_scale:.3f}"
    assert different.max() < 0.75, f"different people score up to {different.max():.3f}"
    assert min(same_tight, same_scale) - different.max() > 0.2, (
        f"same-person {min(same_tight, same_scale):.3f} vs different-person "
        f"{different.max():.3f} leaves no usable margin")


def test_reid_bank_retrieves_the_right_person(rcdl, reid_model):
    """`EmbeddingBank` end to end: enrol each person from one frame, then query
    with a re-cropped version of one of them and check the top match is that
    person. This is the operation a tracker performs on every association, so it
    is worth running against the real model rather than only against numpy."""
    need(rcdl, "ImageEmbedder", "EmbeddingBank")
    det_model = bm.find_model("yolov8n_rk3588.rknn")
    if det_model is None:
        pytest.skip("yolov8n_rk3588.rknn not staged")

    img = bm.load_bgr("bus.jpg")
    det = rcdl.Engine(det_model).detector(model_input="rgb888")
    people = [d for d in rcdl.detect(det, img, "bgr888")
              if rcdl.coco_class_name(d.class_id) == "person"]
    embedder = rcdl.Engine(reid_model).embedder()

    bank = rcdl.EmbeddingBank()
    for i, d in enumerate(people):
        bank.add(np.asarray(rcdl.embed(embedder, img, (d.x1, d.y1, d.x2, d.y2), "bgr888")),
                 f"person{i}")

    hits = 0
    for i, d in enumerate(people):
        w, h = d.x2 - d.x1, d.y2 - d.y1
        q = np.asarray(rcdl.embed(embedder, img,
                                  (d.x1 + 0.03 * w, d.y1 + 0.03 * h,
                                   d.x2 - 0.03 * w, d.y2 - 0.03 * h), "bgr888"))
        best = bank.search(q, 1)[0]
        print(f"  query person{i} -> {best.label} ({best.score:.3f})")
        hits += int(best.label == f"person{i}")
    assert hits == len(people), f"{hits}/{len(people)} queries retrieved the right person"


# --------------------------------------------------------------------------- #
# Tracking — DetectionPipeline + ByteTrack, with and without appearance        #
# --------------------------------------------------------------------------- #
def _panning_sequence(img, frames=14, step=6):
    """The same scene translated a few pixels per frame.

    A synthetic pan rather than a video clip on purpose: it makes the *right
    answer* knowable. Every object is present in every frame and moves by a
    known amount, so any id that appears beyond the object count is association
    failing, not the scene changing. A real clip cannot distinguish the two.
    """
    import cv2
    h, w = img.shape[:2]
    for i in range(frames):
        M = np.float32([[1, 0, -i * step], [0, 1, 0]])
        yield cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


@pytest.mark.parametrize("use_reid", [False, True], ids=["geometry", "reid"])
def test_tracking_holds_identities_across_a_panning_sequence(rcdl, use_reid):
    """`TrackingPipeline` end to end on the board, both of its modes.

    Everything under this test had been verified only by throwaway scripts:
    `test_tracking.py` is a numpy oracle over synthetic boxes, and the ReID half
    could not be run at all until an appearance model existed. What is checked
    here is the one property tracking exists to provide — an id that stays put.

    The failure it catches is id churn: association that silently stops matching
    emits a *new* id every frame while each individual frame still looks
    perfectly reasonable. Counting distinct ids over the sequence is what makes
    that visible. Measured on `bus.jpg` panned 6 px per frame for 14 frames:
    4 tracks, 4 distinct ids, three of them alive for all 14 frames and the
    fourth until it leaves the frame.
    """
    need(rcdl, "TrackingPipeline")
    det_model = bm.require_model("yolov8n_rk3588.rknn")
    reid = None
    if use_reid:
        reid = rcdl.Engine(bm.require_model("osnet_x0_25_msmt17_rk3588.rknn"))

    img = bm.load_bgr("bus.jpg")
    pipeline = rcdl.Engine(det_model).tracker(reid=reid)
    assert pipeline.has_reid == use_reid

    seen, per_frame, embedded = {}, [], []
    for i, frame in enumerate(_panning_sequence(img)):
        tracks = rcdl.track(pipeline, frame, "bgr888")
        per_frame.append(len(tracks))
        embedded.append(pipeline.last_embed_count)
        for t in tracks:
            assert t.track_id > 0, "track ids start at 1"
            assert t.x2 > t.x1 and t.y2 > t.y1, "degenerate track box"
            seen.setdefault(t.track_id, []).append(i)

    lives = sorted((len(v) for v in seen.values()), reverse=True)
    print(f"\n{'reid' if use_reid else 'geometry'}: {len(seen)} distinct ids over 14 frames, "
          f"lifetimes {lives}, tracks per frame {per_frame}")

    assert per_frame[0] >= 3, f"only {per_frame[0]} tracks on the first frame"
    # The scene holds a fixed cast, so the id count must stay close to it. Churn
    # would push this into the dozens.
    assert len(seen) <= per_frame[0] + 2, (
        f"{len(seen)} distinct ids for ~{per_frame[0]} objects — ids are churning")
    assert sum(1 for v in seen.values() if len(v) >= 10) >= 3, (
        f"only {sum(1 for v in seen.values() if len(v) >= 10)} ids survived 10+ frames: {lives}")

    if use_reid:
        # The ReID stage must actually run — and cost what it costs.
        assert max(embedded) > 0, "has_reid is True but no crop was ever embedded"
        assert max(embedded) <= 32, f"embedded {max(embedded)} crops, above the default cap"
        print(f"  crops embedded per frame: {embedded}")
    else:
        assert set(embedded) == {0}, "geometry-only tracking embedded something"


def test_tracking_reset_restarts_the_ids(rcdl):
    """`reset()` is what a caller uses on a stream cut.

    Checking this needs a control, because on an unchanged scene the ids come
    back as 1..N whether reset ran or not — association simply keeps working.
    So the sequence is: track a pan, then cut to a DIFFERENT scene. Without
    reset the new objects are strangers and get fresh ids counting on from the
    old ones; with reset the counter starts over. Running both pipelines over
    the identical frames is what turns "the ids look right" into a test.
    """
    need(rcdl, "TrackingPipeline")
    det_model = bm.require_model("yolov8n_rk3588.rknn")
    img = bm.load_bgr("bus.jpg")
    cut = bm.load_bgr("zidane.jpg")

    kept = rcdl.Engine(det_model).tracker()
    cleared = rcdl.Engine(det_model).tracker()
    for frame in _panning_sequence(img, frames=6):
        rcdl.track(kept, frame, "bgr888")
        rcdl.track(cleared, frame, "bgr888")

    cleared.reset()
    ids_kept = {t.track_id for t in rcdl.track(kept, cut, "bgr888")}
    ids_cleared = {t.track_id for t in rcdl.track(cleared, cut, "bgr888")}
    print(f"\nafter a scene cut — without reset {sorted(ids_kept)}, "
          f"with reset {sorted(ids_cleared)}")

    assert ids_cleared, "no tracks after reset"
    assert min(ids_cleared) == 1, f"ids must restart at 1 after reset, got {sorted(ids_cleared)}"
    assert min(ids_kept) > 1, (
        f"without reset the new scene should get fresh ids counting on from the old "
        f"tracklets, got {sorted(ids_kept)} — the control is not controlling anything")


# --------------------------------------------------------------------------- #
# Sparse local features — XFeat                                                #
# --------------------------------------------------------------------------- #
XFEAT_MODEL = "xfeat_640x480_i8_rk3588.rknn"


def _xfeat(rcdl, **cfg_kw):
    """An extractor on the XFeat model, opened the one way that works.

    `float_inputs=[0]` is not a tuning knob: this model's input is an
    InstanceNorm output, roughly ±3, and the u8 path a quantized model normally
    takes has no negative range at all to put half of that in.
    """
    model = bm.require_model(XFEAT_MODEL)
    cfg = rcdl.XfeatConfig()
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    engine = rcdl.Engine(model, float_inputs=[0])
    return engine, engine.feature_extractor(config=cfg)


def _warped(img, angle=12.0, scale=0.85, tx=25.0, ty=-15.0):
    """A known rotation+scale of `img`, and the 2x3 that produced it."""
    cv2 = pytest.importorskip("cv2")
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT), m


def _at_model_size(rcdl, img, extractor):
    cv2 = pytest.importorskip("cv2")
    return cv2.resize(img, (extractor.input_width, extractor.input_height))


def test_xfeat_matches_survive_a_known_warp(rcdl):
    """The only head here whose ground truth can be MANUFACTURED.

    Rotate and shrink a photograph by a known amount and every correspondence
    has an exact right answer: where the warp says it should land. So this does
    not assert "some matches were found" — a descriptor that returned noise
    would still produce mutual nearest neighbours — it asserts that most of them
    agree with the geometry, to within a few pixels.
    """
    need(rcdl, "FeatureExtractor", "match_features")
    engine, ex = _xfeat(rcdl)
    a = _at_model_size(rcdl, bm.load_bgr("bus.jpg"), ex)
    b, warp = _warped(a)

    fa = rcdl.extract_features(ex, a)
    fb = rcdl.extract_features(ex, b)
    pairs, scores = rcdl.match_features(fa, fb)
    assert len(fa) > 500 and len(fb) > 500, f"only {len(fa)}/{len(fb)} features"
    assert len(pairs) > 200, f"only {len(pairs)} matches"

    pa = fa.xy[pairs[:, 0]]
    pb = fb.xy[pairs[:, 1]]
    proj = (warp[:, :2] @ pa.T).T + warp[:, 2]
    h, w = a.shape[:2]
    inside = ((proj[:, 0] >= 16) & (proj[:, 0] < w - 16) &
              (proj[:, 1] >= 16) & (proj[:, 1] < h - 16))
    err = np.linalg.norm(proj[inside] - pb[inside], axis=1)
    frac = float((err < 3.0).mean())
    print(f"\nXFeat vs a known warp: {len(fa)}+{len(fb)} features, {len(pairs)} matches, "
          f"{int(inside.sum())} in frame, {frac:.1%} within 3 px, median {np.median(err):.2f} px")

    # The float ONNX scores 0.774 on this pair; int8 measured 0.775. The gate is
    # well under both, because it is guarding against a broken contract (wrong
    # channel order, transposed cells, bilinear descriptors) rather than tracking
    # the model's exact quality.
    assert frac > 0.6, f"only {frac:.1%} of matches agree with the known warp"
    assert float(np.median(err)) < 2.0


def test_xfeat_refuses_an_engine_that_would_clip_its_input(rcdl):
    """Opened without `float_inputs`, this model's input is presented as image
    bytes and the negative half of the normalized map is silently clipped to the
    zero point. Keypoints still come out and still look like keypoints, so the
    head has to refuse rather than run."""
    need(rcdl, "FeatureExtractor")
    model = bm.require_model(XFEAT_MODEL)
    engine = rcdl.Engine(model)           # the u8 default
    assert engine.input_dtype(0) == np.uint8
    with pytest.raises(Exception):
        engine.feature_extractor()

    ok = rcdl.Engine(model, float_inputs=[0])
    assert ok.input_dtype(0) == np.float32
    ok.feature_extractor()


def test_xfeat_is_reproducible_frame_to_frame(rcdl):
    """The same frame twice must give byte-identical features.

    This repo has already been bitten by a silently non-deterministic
    preprocessing path (a CPU-filled letterbox border racing the hardware that
    wrote the rest of the tensor), where every unit was deterministic on its own
    and only a full second run showed the difference.
    """
    need(rcdl, "FeatureExtractor")
    _, ex = _xfeat(rcdl)
    img = _at_model_size(rcdl, bm.load_bgr("zidane.jpg"), ex)
    first = rcdl.extract_features(ex, img)
    second = rcdl.extract_features(ex, img)
    assert len(first) == len(second)
    np.testing.assert_array_equal(first.xy, second.xy)
    np.testing.assert_array_equal(first.scores, second.scores)
    np.testing.assert_array_equal(first.descriptors, second.descriptors)


def test_xfeat_descriptors_separate_a_place_from_another_place(rcdl):
    """Matching two DIFFERENT scenes must not produce the same agreement.

    Without this control, "1500 mutual matches" reads as success; mutual nearest
    neighbours exist between any two sets of vectors. The cross-scene pair is
    what shows the number means something.
    """
    need(rcdl, "FeatureExtractor", "match_features")
    _, ex = _xfeat(rcdl)
    bus = _at_model_size(rcdl, bm.load_bgr("bus.jpg"), ex)
    zid = _at_model_size(rcdl, bm.load_bgr("zidane.jpg"), ex)
    same, _ = _warped(bus)

    f_bus = rcdl.extract_features(ex, bus)
    f_same = rcdl.extract_features(ex, same)
    f_zid = rcdl.extract_features(ex, zid)
    self_matches, _ = rcdl.match_features(f_bus, f_same)
    cross_matches, _ = rcdl.match_features(f_bus, f_zid)
    print(f"\nmatches: same scene warped {len(self_matches)}, different scene "
          f"{len(cross_matches)}")
    assert len(self_matches) > 4 * max(len(cross_matches), 1)


# --------------------------------------------------------------------------- #
# Super-resolution — Real-ESRGAN Compact x4                                    #
# --------------------------------------------------------------------------- #
SR_MODEL = "realesr_general_x4v3_128_fp16_rk3588.rknn"
SR_MODEL_I8 = "realesr_general_x4v3_128_i8_rk3588.rknn"


def _sr_pair(rcdl, name="bus.jpg", side=512):
    """(HR crop, LR = HR shrunk by 4) — the original IS the right answer."""
    cv2 = pytest.importorskip("cv2")
    img = bm.load_bgr(name)
    h, w = img.shape[:2]
    s = max(side / h, side / w) * 1.02
    if s > 1:
        img = cv2.resize(img, (int(w * s) + 1, int(h * s) + 1))
        h, w = img.shape[:2]
    hr = np.ascontiguousarray(img[(h - side) // 2:(h - side) // 2 + side,
                                  (w - side) // 2:(w - side) // 2 + side])
    lr = cv2.resize(hr, (side // 4, side // 4), interpolation=cv2.INTER_AREA)
    return hr, lr


def _psnr(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float((d * d).mean())
    return 99.0 if mse <= 0 else 10.0 * np.log10(255.0 * 255.0 / mse)


def _edge_energy(img):
    cv2 = pytest.importorskip("cv2")
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def test_superres_puts_back_detail_a_resize_cannot(rcdl):
    """PSNR cannot judge this model, so the test does not ask it to.

    A perceptually-trained upscaler invents plausible texture instead of the
    blur that minimises squared error, so it scores BELOW bicubic against the
    ground truth while looking obviously better. What it must do is recover the
    high-frequency energy the downscale destroyed — measured here as mean
    gradient magnitude, with bicubic as the floor and the original as the target.
    """
    need(rcdl, "SuperResolver")
    cv2 = pytest.importorskip("cv2")
    model = bm.require_model(SR_MODEL)
    hr, lr = _sr_pair(rcdl)
    sr = rcdl.Engine(model).upscaler()
    assert sr.scale == 4 and sr.tile == 128

    up = rcdl.upscale(sr, lr)
    bicubic = cv2.resize(lr, (up.shape[1], up.shape[0]), interpolation=cv2.INTER_CUBIC)
    e_up, e_bic, e_hr = _edge_energy(up), _edge_energy(bicubic), _edge_energy(hr)
    print(f"\nx4 from {lr.shape[1]}x{lr.shape[0]}: {sr.last_tile_count} tiles, "
          f"edge energy {e_up:.1f} vs bicubic {e_bic:.1f} vs original {e_hr:.1f}; "
          f"PSNR {_psnr(up, hr):.2f} dB vs bicubic {_psnr(bicubic, hr):.2f} dB")

    assert up.shape == (hr.shape[0], hr.shape[1], 3)
    assert e_up > 1.2 * e_bic, "the upscale is no sharper than a resize"
    assert e_up > 0.7 * e_hr, "most of the detail the downscale destroyed is still missing"
    # A scrambled tiling or a swapped channel order still produces *an* image;
    # it does not survive being shrunk back to the input it came from.
    back = cv2.resize(up, (lr.shape[1], lr.shape[0]), interpolation=cv2.INTER_AREA)
    assert _psnr(back, lr) > 20.0, "the result is not consistent with its own input"


def test_superres_colours_are_not_swapped(rcdl):
    """The model is RGB and every image entry point here is BGR. Getting that
    backwards costs nothing in sharpness or size — the picture just comes out
    with its channels exchanged, which no geometric check would notice."""
    need(rcdl, "SuperResolver")
    _, lr = _sr_pair(rcdl)
    sr = rcdl.Engine(bm.require_model(SR_MODEL)).upscaler()
    up = rcdl.upscale(sr, lr)
    cv2 = pytest.importorskip("cv2")
    small = cv2.resize(up, (lr.shape[1], lr.shape[0]), interpolation=cv2.INTER_AREA)
    per_channel = [_psnr(small[..., c], lr[..., c]) for c in range(3)]
    swapped = [_psnr(small[..., 2 - c], lr[..., c]) for c in range(3)]
    print(f"\nper-channel PSNR straight {['%.1f' % v for v in per_channel]}, "
          f"B<->R swapped {['%.1f' % v for v in swapped]}")
    assert min(per_channel) > min(swapped) + 3.0


def test_superres_tiling_leaves_no_seam(rcdl):
    """Two tiles disagree slightly where their receptive fields were truncated.
    Cross-fading spreads that over a band; butt-joining concentrates it on one
    column, which is what the eye finds.

    This needs a source LARGER than one tile or it measures nothing at all —
    with a single tile both settings return the same image. The seam columns are
    known (butt-jointed tiles meet every `tile * scale` pixels), so the test
    compares the jump exactly there against the jump everywhere else.
    """
    need(rcdl, "SuperResolver")
    cv2 = pytest.importorskip("cv2")
    engine = rcdl.Engine(bm.require_model(SR_MODEL))
    probe = engine.upscaler()
    lr = cv2.resize(bm.load_bgr("zidane.jpg"), (probe.tile * 2 + 40, probe.tile + 30))

    butted_up = engine.upscaler(overlap=0)
    butted = rcdl.upscale(butted_up, lr)
    blended_up = engine.upscaler(overlap=16)
    blended = rcdl.upscale(blended_up, lr)
    assert butted_up.last_tile_count >= 4, "the source has to span several tiles"

    seams = [x for x in range(1, butted.shape[1])
             if x % (probe.tile * probe.scale) == 0]

    def seam_ratio(img):
        col = img.astype(np.float64).mean(axis=2)
        jump = np.abs(np.diff(col, axis=1)).mean(axis=0)   # per boundary column
        elsewhere = np.delete(jump, [s - 1 for s in seams]).mean()
        return float(jump[[s - 1 for s in seams]].mean() / (elsewhere + 1e-6))

    b, u = seam_ratio(blended), seam_ratio(butted)
    print(f"\nseam columns {seams}, jump vs elsewhere: blended {b:.2f}x, "
          f"butt-jointed {u:.2f}x ({butted_up.last_tile_count} tiles)")
    assert u > 1.15, "butt-jointed tiles left no measurable seam — nothing is being tested"
    assert b < u, "the cross-fade did not reduce the seam it exists to remove"


def test_superres_int8_is_the_faster_lower_fidelity_build(rcdl):
    """Both builds are in the registry and they are not interchangeable.

    int8 is ~1.8x faster and does not reproduce the float model: it disagrees by
    about 31 dB and over-sharpens past the original. This is the measurement the
    choice is made on, so it is pinned rather than described.
    """
    need(rcdl, "SuperResolver")
    fp16 = bm.find_model(SR_MODEL)
    i8 = bm.find_model(SR_MODEL_I8)
    if not fp16 or not i8:
        pytest.skip("both the fp16 and the int8 super-resolution builds are needed")
    hr, lr = _sr_pair(rcdl)
    a = rcdl.upscale(rcdl.Engine(fp16).upscaler(), lr)
    b = rcdl.upscale(rcdl.Engine(i8).upscaler(), lr)
    agree = _psnr(b, a)
    print(f"\nint8 vs fp16 output: {agree:.2f} dB; edge energy fp16 {_edge_energy(a):.1f}, "
          f"int8 {_edge_energy(b):.1f}, original {_edge_energy(hr):.1f}")
    assert 25.0 < agree < 45.0, (
        "int8 is expected to differ from the float build by roughly 31 dB — "
        "a much higher number means the builds are no longer what this test describes")
    assert _edge_energy(b) > _edge_energy(a), "the int8 build's extra edge energy is gone"


# --------------------------------------------------------------------------- #
# Dense optical flow — NeuFlow v2, and the custom operator under it            #
# --------------------------------------------------------------------------- #
FLOW_MODEL = "neuflow_v2_512x384_fp16_rk3588.rknn"


def _flow_window_pair(rcdl, estimator, dx, dy, name="bus.jpg"):
    """Two windows of one photograph, offset by (dx, dy).

    A window that moves right sees the world move LEFT, so the true flow is
    (-dx, -dy) everywhere except the band where content entered or left.
    """
    cv2 = pytest.importorskip("cv2")
    img = bm.load_bgr(name)
    w, h = estimator.input_width, estimator.input_height
    need_w, need_h = w + abs(dx) + 4, h + abs(dy) + 4
    if img.shape[1] < need_w or img.shape[0] < need_h:
        s = max(need_w / img.shape[1], need_h / img.shape[0]) * 1.05
        img = cv2.resize(img, (int(img.shape[1] * s) + 1, int(img.shape[0] * s) + 1))
    x0 = (img.shape[1] - w - abs(dx)) // 2
    y0 = (img.shape[0] - h - abs(dy)) // 2
    a = np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w])
    b = np.ascontiguousarray(img[y0 + dy:y0 + dy + h, x0 + dx:x0 + dx + w])
    return a, b, -float(dx), -float(dy)


@pytest.mark.parametrize("dx,dy", [(0, 0), (2, 0), (8, 0), (6, 10)])
def test_flow_measures_a_known_shift(rcdl, dx, dy):
    """Flow is the one task whose ground truth can be manufactured exactly: move
    a window across a photograph and the correct field is that constant.

    The float ONNX scores 0.026 px on a static pair and 0.03-0.10 px on shifts
    of 2-16; this build has to stay in that neighbourhood, not merely produce a
    field that points the right way.
    """
    need(rcdl, "OpticalFlowEstimator")
    model = bm.require_model(FLOW_MODEL)
    est = rcdl.Engine(model).flow_estimator()
    a, b, gu, gv = _flow_window_pair(rcdl, est, dx, dy)

    field = rcdl.estimate_flow(est, a, b)
    assert field.shape == (est.input_height, est.input_width, 2)
    m = 48                                   # the border has no correct answer
    inner = field[m:-m, m:-m]
    err = np.hypot(inner[..., 0] - gu, inner[..., 1] - gv)
    print(f"\nshift ({dx},{dy}): mean vector ({inner[..., 0].mean():.3f}, "
          f"{inner[..., 1].mean():.3f}) expect ({gu:.1f}, {gv:.1f}), EPE {err.mean():.4f} px")
    assert err.mean() < 0.5, f"endpoint error {err.mean():.3f} px against a known shift"
    assert abs(inner[..., 0].mean() - gu) < 0.2
    assert abs(inner[..., 1].mean() - gv) < 0.2


def test_flow_follows_a_rotation_not_just_a_translation(rcdl):
    """A uniform shift cannot tell a correct field from one with a coordinate
    convention wrong inside the correlation lookup — every pixel has the same
    answer, so a systematic distortion of the sampling grid still averages out.
    A rotation does not: the true field varies across the frame, and its value
    at each pixel is still known exactly.
    """
    need(rcdl, "OpticalFlowEstimator")
    cv2 = pytest.importorskip("cv2")
    est = rcdl.Engine(bm.require_model(FLOW_MODEL)).flow_estimator()
    w, h = est.input_width, est.input_height
    img = cv2.resize(bm.load_bgr("bus.jpg"), (w, h))

    theta = np.deg2rad(3.0)
    ca, sa = np.cos(theta), np.sin(theta)
    cx, cy = w / 2.0, h / 2.0
    mat = np.array([[ca, -sa, cx - ca * cx + sa * cy],
                    [sa, ca, cy - sa * cx - ca * cy]], np.float64)
    rotated = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)

    field = rcdl.estimate_flow(est, img, rotated)
    ys, xs = np.mgrid[0:h, 0:w]
    gt_u = mat[0, 0] * xs + mat[0, 1] * ys + mat[0, 2] - xs
    gt_v = mat[1, 0] * xs + mat[1, 1] * ys + mat[1, 2] - ys
    m = 64
    err = np.hypot(field[m:-m, m:-m, 0] - gt_u[m:-m, m:-m],
                   field[m:-m, m:-m, 1] - gt_v[m:-m, m:-m])
    speeds = np.hypot(gt_u, gt_v)[m:-m, m:-m]
    span = float(speeds.max() - speeds.min())
    print(f"\n3 deg rotation: true speed varies over {span:.1f} px across the frame, "
          f"EPE {err.mean():.3f} px (median {np.median(err):.3f})")
    assert span > 5.0, "the control is not controlling: this rotation is nearly a translation"
    # The float ONNX scores 0.145 px here and the toolkit's simulator agrees with
    # it to three decimals; the NPU's fp16 arithmetic costs the rest, and that
    # gap is the point of docs/MODELS.md's note about the simulator. The gate is
    # set to catch a broken field (which would be metres, not tenths of a pixel),
    # not to track the model's precision.
    assert err.mean() < 1.2, f"endpoint error {err.mean():.3f} px on a non-uniform field"


def test_flow_needs_the_custom_operator_and_says_so(rcdl):
    """This model contains a `GridSample`, which librknnrt 2.3.2 implements
    neither on the NPU nor on its own CPU fallback path. It was converted with
    that node declared as a custom operator and RCDL registers the kernel at
    construction; without the registration the graph loads and then fails at
    rknn_run. Both halves are asserted, because "it works" is only interesting
    next to "and here is what it needs".
    """
    need(rcdl, "OpticalFlowEstimator")
    model = bm.require_model(FLOW_MODEL)
    est = rcdl.Engine(model).flow_estimator()
    a, b, _, _ = _flow_window_pair(rcdl, est, 8, 0)
    rcdl.estimate_flow(est, a, b)                     # registered: fine

    bare = rcdl.Engine(model, custom_ops=False)
    bare_est = bare.flow_estimator()
    with pytest.raises(Exception):
        rcdl.estimate_flow(bare_est, a, b)


def test_flow_is_reproducible(rcdl):
    """Same pair twice, byte-identical — the CPU kernel runs OpenMP-parallel over
    the batch and writes into the runtime's own buffers, which is exactly where a
    race would hide."""
    need(rcdl, "OpticalFlowEstimator")
    est = rcdl.Engine(bm.require_model(FLOW_MODEL)).flow_estimator()
    a, b, _, _ = _flow_window_pair(rcdl, est, 6, 10)
    first = rcdl.estimate_flow(est, a, b)
    second = rcdl.estimate_flow(est, a, b)
    np.testing.assert_array_equal(first, second)


# --------------------------------------------------------------------------- #
# Promptable segmentation — EdgeSAM                                            #
# --------------------------------------------------------------------------- #
SAM_ENCODER = "edge_sam_3x_encoder_fp16_rk3588.rknn"
SAM_DECODER = "edge_sam_3x_decoder_fp16_rk3588.rknn"


def _sam(rcdl, **kw):
    enc = rcdl.Engine(bm.require_model(SAM_ENCODER))
    dec = rcdl.Engine(bm.require_model(SAM_DECODER))
    return enc, dec, enc.prompt_segmenter(dec, **kw)


def _set_image(rcdl, sam, img):
    flat = np.ascontiguousarray(img).reshape(-1)
    sam.set_image(flat, img.shape[1], img.shape[0], "bgr888")


def test_sam_box_prompt_agrees_with_the_instance_segmenter(rcdl):
    """The control that makes this head's output mean something.

    A mask that covers the prompt box is easy to produce and easy to fake — a
    model returning the box itself would score well on every "is it in the right
    place" check. So the mask is compared against a DIFFERENT model's mask for
    the same object: yolov8n-seg's bus. Two unrelated networks agreeing on a
    silhouette is evidence; either one agreeing with its own prompt is not.
    """
    need(rcdl, "PromptableSegmenter")
    seg_model = bm.require_model("yolov8n-seg_rk3588.rknn")
    img = bm.load_bgr("bus.jpg")

    inst = rcdl.segment_instances(rcdl.Engine(seg_model).instance_segmenter(), img)
    bus = max((d for d in inst if rcdl.coco_class_name(d.class_id) == "bus"),
              key=lambda d: d.score, default=None)
    if bus is None:
        pytest.skip("the instance segmenter found no bus to compare against")
    ref = np.asarray(bus.mask).astype(bool)

    _, _, sam = _sam(rcdl)
    _set_image(rcdl, sam, img)
    got = sam.box(bus.x1, bus.y1, bus.x2, bus.y2)
    m = np.asarray(got.mask).astype(bool)

    inter = np.logical_and(m, ref).sum()
    union = np.logical_or(m, ref).sum()
    iou = float(inter) / float(union) if union else 0.0
    box_area = (bus.x2 - bus.x1) * (bus.y2 - bus.y1)
    print(f"\nSAM box prompt vs yolov8n-seg on the bus: IoU {iou:.3f}, "
          f"SAM {100 * got.area:.1f}% of the frame, score {got.score:.3f}")

    assert iou > 0.75, f"the two models disagree about the bus (IoU {iou:.3f})"
    # And it is not simply returning the prompt: a filled box would cover the
    # whole rectangle, and a bus does not.
    filled = float(m.sum()) / box_area
    assert filled < 0.92, f"the mask fills {filled:.0%} of the prompt box — that is the box"


def test_sam_point_prompt_finds_the_thing_under_the_click(rcdl):
    """A click inside a person must return that person: a mask containing the
    click, no larger than a person, and centred near them."""
    need(rcdl, "PromptableSegmenter")
    det_model = bm.require_model("yolov8n_rk3588.rknn")
    img = bm.load_bgr("bus.jpg")
    dets = rcdl.detect(rcdl.Engine(det_model).detector(), img)
    people = [d for d in dets if rcdl.coco_class_name(d.class_id) == "person"]
    if not people:
        pytest.skip("no person detected to click on")
    p = max(people, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))
    cx, cy = (p.x1 + p.x2) / 2.0, (p.y1 + p.y2) / 2.0

    _, _, sam = _sam(rcdl)
    _set_image(rcdl, sam, img)
    got = sam.point(cx, cy)
    m = np.asarray(got.mask).astype(bool)
    print(f"\nclick at ({cx:.0f},{cy:.0f}) inside a person box "
          f"[{p.x1:.0f} {p.y1:.0f} {p.x2:.0f} {p.y2:.0f}] -> {100 * got.area:.2f}% of the "
          f"frame, score {got.score:.3f}, bbox {got.bbox}")

    assert m[int(cy), int(cx)], "the mask does not contain the point that produced it"
    inside = m[int(p.y1):int(p.y2), int(p.x1):int(p.x2)].sum()
    assert inside / max(m.sum(), 1) > 0.8, "most of the mask is outside the person clicked"


def test_sam_encodes_once_and_prompts_many_times(rcdl):
    """The split is the reason this head is usable at all, so it is asserted:
    a second prompt on the same image must cost a fraction of the first
    set_image, and must not silently re-encode."""
    need(rcdl, "PromptableSegmenter")
    import time
    img = bm.load_bgr("bus.jpg")
    _, _, sam = _sam(rcdl)

    t0 = time.perf_counter()
    _set_image(rcdl, sam, img)
    t1 = time.perf_counter()
    a = sam.box(36, 230, 799, 771)
    t2 = time.perf_counter()
    b = sam.box(36, 230, 799, 771)
    t3 = time.perf_counter()
    encode_ms, first_ms, second_ms = (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3
    print(f"\nencode {encode_ms:.0f} ms, prompt {first_ms:.0f} ms then {second_ms:.0f} ms")

    assert second_ms < encode_ms, "a repeat prompt costs as much as encoding — it re-encoded"
    np.testing.assert_array_equal(np.asarray(a.mask), np.asarray(b.mask))
    assert a.score == b.score


def test_sam_returns_several_nestings_for_one_prompt(rcdl):
    """SAM answers an ambiguous prompt with several masks and its own quality
    estimate for each. The head must expose all of them and hand back the
    highest-scoring one by default."""
    need(rcdl, "PromptableSegmenter")
    img = bm.load_bgr("bus.jpg")
    _, _, sam = _sam(rcdl)
    _set_image(rcdl, sam, img)
    best = sam.point(img.shape[1] / 2.0, img.shape[0] / 2.0)
    every = sam.masks()
    print(f"\nnestings for one click: " +
          ", ".join(f"{100 * m.area:.1f}% @ {m.score:.3f}" for m in every))

    assert len(every) >= 3
    assert every[0].score >= every[-1].score, "masks() is not best-first"
    assert best.score == pytest.approx(every[0].score)
    areas = sorted(m.area for m in every)
    assert areas[-1] > areas[0] * 1.2, "every nesting is the same size — the ambiguity is gone"


def test_sam_refuses_an_encoder_that_takes_image_bytes(rcdl):
    """The int8 encoder is in the recipe but not in the registry: it keeps the
    shape of large objects and loses small ones entirely. It is also a quantized
    model, so its input is image bytes rather than the float tensor this head
    writes — and the head says so instead of feeding it the wrong thing."""
    need(rcdl, "PromptableSegmenter")
    i8 = bm.find_model("edge_sam_3x_encoder_i8_rk3588.rknn")
    if not i8:
        pytest.skip("no int8 encoder staged (it is deliberately not in the registry)")
    dec = rcdl.Engine(bm.require_model(SAM_DECODER))
    with pytest.raises(Exception):
        rcdl.Engine(i8).prompt_segmenter(dec)


# --------------------------------------------------------------------------- #
# Whole-body pose — RTMW, 133 keypoints                                        #
# --------------------------------------------------------------------------- #
WHOLEBODY_MODEL = "rtmw_s_133_256x192_fp16_rk3588.rknn"


def _biggest_person(rcdl, img):
    det = rcdl.Engine(bm.require_model("yolov8n_rk3588.rknn")).detector()
    people = [d for d in rcdl.detect(det, img)
              if rcdl.coco_class_name(d.class_id) == "person"]
    if not people:
        pytest.skip("no person detected")
    return max(people, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))


def test_wholebody_agrees_with_the_plain_pose_head_on_the_body_joints(rcdl):
    """The control that makes 133 keypoints mean something.

    The first 17 of the whole-body layout ARE the COCO body joints, in order, so
    a completely different model — yolov8n-pose, bottom-up, trained separately —
    can be asked the same question. Two networks landing on the same shoulders
    and wrists is evidence; a skeleton that merely looks like a person is not.
    """
    need(rcdl, "WholeBodyEstimator")
    img = bm.load_bgr("bus.jpg")
    person = _biggest_person(rcdl, img)

    pose_model = bm.require_model("yolov8n-pose_rk3588.rknn")
    poses = rcdl.estimate_pose(rcdl.Engine(pose_model).pose_estimator(), img)
    if not poses:
        pytest.skip("the pose head found nobody to compare against")
    # The same person: the pose detection whose box overlaps the detector's most.
    def overlap(p):
        ix = max(0.0, min(p.box.x2, person.x2) - max(p.box.x1, person.x1))
        iy = max(0.0, min(p.box.y2, person.y2) - max(p.box.y1, person.y1))
        return ix * iy
    ref = max(poses, key=overlap)
    if overlap(ref) <= 0:
        pytest.skip("the two heads found no person in common")

    wb = rcdl.Engine(bm.require_model(WHOLEBODY_MODEL)).wholebody_estimator()
    kp = rcdl.estimate_wholebody(wb, img, (person.x1, person.y1, person.x2, person.y2))
    assert kp.shape == (133, 3)

    diag = float(np.hypot(person.x2 - person.x1, person.y2 - person.y1))
    errs = []
    for i in range(17):
        a, b = kp[i], ref.keypoints[i]
        if kp[i, 2] < 0.3 or b.score < 0.5:
            continue
        errs.append(np.hypot(a[0] - b.x, a[1] - b.y))
    print(f"\nwhole-body vs yolov8n-pose on {len(errs)} shared body joints: "
          f"median {np.median(errs):.1f} px, max {max(errs):.1f} px "
          f"(person diagonal {diag:.0f} px)")
    assert len(errs) >= 10, "too few joints in common to compare"
    assert np.median(errs) < 0.05 * diag, "the two heads disagree about where the body is"


def test_wholebody_face_and_hands_are_where_a_body_puts_them(rcdl):
    """133 points is only useful if the extra 116 are in the right places: the
    face cluster on the head, each hand cluster at its own wrist. This is the
    check that catches a transposed or mis-sliced layout, which a body-only
    comparison cannot see."""
    need(rcdl, "WholeBodyEstimator")
    img = bm.load_bgr("bus.jpg")
    person = _biggest_person(rcdl, img)
    wb = rcdl.Engine(bm.require_model(WHOLEBODY_MODEL)).wholebody_estimator()
    kp = rcdl.estimate_wholebody(wb, img, (person.x1, person.y1, person.x2, person.y2))

    def centre(part):
        b, e = rcdl.body_part_range(part)
        sel = kp[b:e][kp[b:e, 2] >= 0.3]
        if len(sel) == 0:
            pytest.skip(f"no confident {rcdl.body_part_name(part)} keypoints")
        return sel[:, :2].mean(axis=0), sel

    face, face_pts = centre(rcdl.BodyPart.FACE)
    lhand, _ = centre(rcdl.BodyPart.LEFT_HAND)
    rhand, _ = centre(rcdl.BodyPart.RIGHT_HAND)
    nose, l_wrist, r_wrist = kp[0, :2], kp[9, :2], kp[10, :2]
    diag = float(np.hypot(person.x2 - person.x1, person.y2 - person.y1))
    print(f"\nface centre {face.round(0)} vs nose {nose.round(0)}; "
          f"hand centres {lhand.round(0)} / {rhand.round(0)} vs wrists "
          f"{l_wrist.round(0)} / {r_wrist.round(0)} (diagonal {diag:.0f} px)")

    assert np.hypot(*(face - nose)) < 0.12 * diag, "the face cluster is not on the head"
    assert np.hypot(*(lhand - l_wrist)) < 0.15 * diag, "the left hand is not at the left wrist"
    assert np.hypot(*(rhand - r_wrist)) < 0.15 * diag, "the right hand is not at the right wrist"
    # And the face is a cluster, not a collapsed point: 68 landmarks spread over
    # a head, which is what distinguishes a decode from a constant.
    spread = float(np.linalg.norm(face_pts[:, :2] - face, axis=1).mean())
    assert spread > 0.01 * diag, "the 68 face points collapsed onto one another"


def test_wholebody_crop_padding_changes_what_the_model_sees(rcdl):
    """The 1.25 padding is part of the model's contract, not a preference: it is
    what keeps hands and feet inside the crop. Estimating the same person with no
    padding must move joints — if it does not, the padding is being ignored."""
    need(rcdl, "WholeBodyEstimator")
    img = bm.load_bgr("bus.jpg")
    person = _biggest_person(rcdl, img)
    box = (person.x1, person.y1, person.x2, person.y2)
    engine = rcdl.Engine(bm.require_model(WHOLEBODY_MODEL))

    padded = rcdl.estimate_wholebody(engine.wholebody_estimator(padding=1.25), img, box)
    tight = rcdl.estimate_wholebody(engine.wholebody_estimator(padding=1.0), img, box)
    both = (padded[:, 2] >= 0.3) & (tight[:, 2] >= 0.3)
    moved = np.hypot(padded[both, 0] - tight[both, 0], padded[both, 1] - tight[both, 1])
    print(f"\npadding 1.25 vs 1.0: {both.sum()} joints in common, median move "
          f"{np.median(moved):.2f} px, mean score {padded[:, 2].mean():.3f} vs "
          f"{tight[:, 2].mean():.3f}")
    assert np.median(moved) > 0.5, "the padding setting had no effect on the crop"
    assert padded[:, 2].mean() >= tight[:, 2].mean() - 0.05, (
        "the model's own trained padding scored materially worse than a tight crop")


def test_wholebody_is_reproducible(rcdl):
    need(rcdl, "WholeBodyEstimator")
    img = bm.load_bgr("bus.jpg")
    person = _biggest_person(rcdl, img)
    wb = rcdl.Engine(bm.require_model(WHOLEBODY_MODEL)).wholebody_estimator()
    box = (person.x1, person.y1, person.x2, person.y2)
    a = rcdl.estimate_wholebody(wb, img, box)
    b = rcdl.estimate_wholebody(wb, img, box)
    np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
# Open-vocabulary detection — YOLOE-11s with a vocabulary baked in             #
# --------------------------------------------------------------------------- #
YOLOE_COCO = "yoloe_11s_coco80_rk3588.rknn"
YOLOE_STREET = "yoloe_11s_streetwear_rk3588.rknn"


def _labels_for(rcdl, model_path):
    """The `.labels.txt` staged beside a model. Open-vocabulary class ids mean
    nothing without it."""
    import os
    p = model_path[: -len(".rknn")] + ".labels.txt"
    if not os.path.isfile(p):
        pytest.skip(f"labels file not staged beside the model: {p}")
    return rcdl.LabelMap.from_file(p)


def _detect_with(rcdl, model_name, img, conf=0.25):
    model = bm.require_model(model_name)
    lm = _labels_for(rcdl, model)
    engine = rcdl.Engine(model)
    pipe = rcdl.DetectionPipeline(engine._e, num_classes=len(lm), conf_thresh=conf)
    dets = pipe.process(np.ascontiguousarray(img).reshape(-1), img.shape[1], img.shape[0],
                        "bgr888")
    return lm, dets


def _iou(a, b):
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / (union + 1e-9)


def test_an_open_vocabulary_model_decodes_through_the_ordinary_detector(rcdl):
    """YOLOE's whole point is that the class axis is chosen at CONVERSION time —
    the CLIP text encoder runs on the host and is folded into the classification
    conv, so what reaches the board is a plain 9-output LTRB head. This asserts
    exactly that: the standard DetectionPipeline, no new decode, reads the same
    scene an ordinary COCO detector reads. Measured: bus 0.946 and four people,
    the bus box within IoU 0.9 of yolov8n's."""
    need(rcdl, "LabelMap", "DetectionPipeline")
    img = bm.load_bgr("bus.jpg")
    lm, dets = _detect_with(rcdl, YOLOE_COCO, img)
    assert len(lm) == 80

    counts = {}
    for d in dets:
        counts[lm.name(d.class_id)] = counts.get(lm.name(d.class_id), 0) + 1
    print(f"\nYOLOE(coco80): {counts}")
    assert counts.get("bus", 0) == 1, counts
    assert counts.get("person", 0) >= 4, counts

    # Cross-model: a separately trained closed-vocabulary detector on the same
    # frame. Agreement here cannot come from the decode being self-consistent.
    v8 = rcdl.Engine(bm.require_model("yolov8n_rk3588.rknn"))
    ref = rcdl.DetectionPipeline(v8._e, conf_thresh=0.25).process(
        np.ascontiguousarray(img).reshape(-1), img.shape[1], img.shape[0], "bgr888")
    ref_bus = [d for d in ref if rcdl.coco_class_name(d.class_id) == "bus"]
    got_bus = [d for d in dets if lm.name(d.class_id) == "bus"]
    assert ref_bus and got_bus
    overlap = _iou(ref_bus[0], got_bus[0])
    print(f"  bus box vs yolov8n: IoU {overlap:.3f}")
    assert overlap > 0.85, "the two models disagree about where the bus is"


def test_a_prompt_coco_has_no_class_for_is_found_where_it_belongs(rcdl):
    """The build that shows the vocabulary is load-bearing rather than
    decorative. "sneakers" is not a COCO class and cannot be expressed by any
    class id of the model above — yet this build finds four pairs, and the check
    is geometric and cross-model: every pair must sit in the bottom fifth of a
    person box found by yolov8n. A vocabulary that was being ignored would give
    either nothing or boxes unrelated to feet."""
    need(rcdl, "LabelMap", "DetectionPipeline")
    img = bm.load_bgr("bus.jpg")
    lm, dets = _detect_with(rcdl, YOLOE_STREET, img)
    assert "sneakers" in list(lm.names)
    assert "sneakers" not in bm.COCO_NAMES, "pick a prompt COCO cannot express"

    shoes = [d for d in dets if lm.name(d.class_id) == "sneakers"]
    print(f"\nYOLOE({len(lm)} prompts): {len(shoes)} sneakers, "
          f"scores {[round(d.score, 3) for d in shoes]}")
    assert len(shoes) >= 3, "the open-vocabulary prompt found nothing"

    v8 = rcdl.Engine(bm.require_model("yolov8n_rk3588.rknn"))
    people = [d for d in rcdl.DetectionPipeline(v8._e, conf_thresh=0.25).process(
        np.ascontiguousarray(img).reshape(-1), img.shape[1], img.shape[0], "bgr888")
        if rcdl.coco_class_name(d.class_id) == "person"]
    assert people
    for s in shoes:
        cx, cy = (s.x1 + s.x2) / 2, (s.y1 + s.y2) / 2
        at_feet = [p for p in people
                   if p.x1 - 10 <= cx <= p.x2 + 10 and cy >= p.y1 + 0.75 * (p.y2 - p.y1)]
        assert at_feet, (f"a 'sneakers' box at ({cx:.0f},{cy:.0f}) is not at the bottom "
                         f"of any person yolov8n found")


def test_the_labels_file_is_checked_against_the_model(rcdl):
    """A labels file from a different build moves no box and changes no score —
    it only renames every result. require_size is the only thing standing
    between the two vocabularies here, so pin that it fires."""
    need(rcdl, "LabelMap")
    coco = bm.require_model(YOLOE_COCO)
    street = bm.require_model(YOLOE_STREET)
    coco_labels = _labels_for(rcdl, coco)
    street_labels = _labels_for(rcdl, street)

    def declared(model_path):
        """What the MODEL says its class axis is, resolved from its output
        signature with NO hint from us. Asking with a hint would only confirm
        the hint — resolveYoloHead picks the branch matching whatever it is
        told, which would make this whole test a tautology."""
        return rcdl.yolo_head_classes(rcdl.Engine(model_path)._e)

    assert declared(coco) == 80 and declared(street) == 6
    for model, labels in ((coco, coco_labels), (street, street_labels)):
        labels.require_size(declared(model))            # matching pair: no throw

    # And the hazard the plain "resolve with a hint" check cannot see: 64 names
    # against the 80-class model. There IS a 64-channel branch — the DFL box
    # head — so the claim "fits", and the 80-channel class branch gets
    # reinterpreted as a box of reg_max 20, which no export produces.
    with pytest.raises(Exception):
        rcdl.yolo_head_classes(rcdl.Engine(coco)._e, 64)
    # A real claim still resolves.
    assert rcdl.yolo_head_classes(rcdl.Engine(coco)._e, 80) == 80
    assert rcdl.yolo_head_classes(rcdl.Engine(street)._e, 6) == 6

    print(f"\ncoco labels: {len(coco_labels)}, streetwear model classes: {declared(street)}")
    with pytest.raises(Exception):                      # 80 names, 6-class model
        coco_labels.require_size(declared(street))
    with pytest.raises(Exception):                      # and the other way round
        street_labels.require_size(declared(coco))
    # And the pairing the library does for you must fail the same way.
    with pytest.raises(Exception):
        rcdl.Engine(street).label_map(coco[: -len(".rknn")] + ".labels.txt")


# --------------------------------------------------------------------------- #
# Panoptic driving — YOLOP: an anchor-based detector + two masks, one inference #
# --------------------------------------------------------------------------- #
YOLOP_MODEL = "yolop_cut_640_i8_rk3588.rknn"
# The prior set is part of the model, not a tuning knob — see docs/MODELS.md.
YOLOP_ANCHORS = [[(3, 9), (5, 11), (4, 20)],
                 [(7, 18), (6, 39), (12, 31)],
                 [(19, 50), (38, 81), (68, 157)]]


def _yolop(rcdl, conf=0.35, anchors=None):
    cfg = rcdl.AnchorDetectConfig()
    cfg.num_classes = 1                       # vehicles
    cfg.conf_thresh = conf
    cfg.strides = [8, 16, 32]
    cfg.anchors = [[rcdl.Anchor(w, h) for w, h in scale]
                   for scale in (anchors if anchors is not None else YOLOP_ANCHORS)]
    engine = rcdl.Engine(bm.require_model(YOLOP_MODEL))
    return engine, cfg


def _yolop_run(rcdl, img, conf=0.35, anchors=None):
    """One inference, three decoders — which is the shape being tested."""
    engine, cfg = _yolop(rcdl, conf, anchors)
    det = rcdl.AnchorDetector(engine._e, cfg, 0)
    drive = rcdl.Segmenter(engine._e, num_classes=2, output_index=3)
    lane = rcdl.Segmenter(engine._e, num_classes=2, output_index=4)
    flat = np.ascontiguousarray(img).reshape(-1)
    drivable = drive.process(flat, img.shape[1], img.shape[0], "bgr888")
    lb = drive.letterbox                       # the geometry that inference used
    return det.postprocess(lb), drivable, lane.postprocess(lb), engine


def test_panoptic_driving_runs_three_heads_off_one_inference(rcdl):
    """The multi-head shape nothing else here covers: an anchor-BASED detector
    reading three raw head tensors plus two segmentation masks, all off a single
    inference. Measured on this street frame: 18 vehicle boxes, drivable area
    22% of the pixels and lane lines 1.8%."""
    need(rcdl, "AnchorDetector", "Segmenter")
    img = bm.load_bgr("cityscapes.png")
    engine = rcdl.Engine(bm.require_model(YOLOP_MODEL))
    assert engine.num_outputs == 5, "3 raw detection heads + drivable + lane"

    dets, drivable, lane, _ = _yolop_run(rcdl, img)
    print(f"\nYOLOP: {len(dets)} boxes")
    assert len(dets) >= 8, "an empty detection head is the broken-export symptom"
    for d in dets:
        assert d.class_id == 0
        assert d.x2 > d.x1 and d.y2 > d.y1
        assert 0 <= d.x1 <= img.shape[1] and 0 <= d.y1 <= img.shape[0]

    for name, mask in (("drivable", drivable), ("lane", lane)):
        labels = np.asarray(mask.labels)
        assert labels.shape == img.shape[:2], "the mask is not projected onto the frame"
        assert set(np.unique(labels)) <= {0, 1}, f"{name} is a 2-class mask"
        on = labels.astype(bool)
        assert on.any(), f"{name} came back empty"
        centroid = float(np.nonzero(on)[0].mean()) / labels.shape[0]
        print(f"  {name}: {100 * on.mean():.2f}% of the frame, centroid at "
              f"{centroid:.2f} of the height")
        # Both are road surface seen from a car: they belong BELOW the horizon.
        # A transposed or channel-swapped mask still covers a plausible area,
        # so area alone would not catch it — where it sits does.
        assert centroid > 0.55, f"{name} is not concentrated in the lower frame"
    assert np.asarray(drivable.labels).astype(bool)[: img.shape[0] // 3].mean() < 0.02, (
        "drivable area above the horizon")


def test_yolop_finds_the_vehicles_another_model_finds(rcdl):
    """Cross-model, because "18 boxes" on its own is not evidence. yolov8n is
    separately trained on a different dataset, so agreement is about the scene
    rather than about this decode being self-consistent. Measured: all 7 of
    yolov8n's vehicles matched, best IoUs 0.65–0.95."""
    need(rcdl, "AnchorDetector")
    img = bm.load_bgr("cityscapes.png")
    dets, _, _, _ = _yolop_run(rcdl, img)

    v8 = rcdl.Engine(bm.require_model("yolov8n_rk3588.rknn"))
    ref = [d for d in rcdl.DetectionPipeline(v8._e, conf_thresh=0.3).process(
        np.ascontiguousarray(img).reshape(-1), img.shape[1], img.shape[0], "bgr888")
        if rcdl.coco_class_name(d.class_id) in ("car", "bus", "truck", "motorcycle", "bicycle")]
    assert ref, "yolov8n found no vehicles — wrong reference frame"
    best = [max((_iou(a, b) for b in dets), default=0.0) for a in ref]
    print(f"\n{sum(v > 0.5 for v in best)}/{len(ref)} of yolov8n's vehicles matched; "
          f"best IoUs {[round(v, 2) for v in sorted(best, reverse=True)]}")
    assert sum(v > 0.5 for v in best) >= len(ref) - 1


def test_the_anchor_priors_are_part_of_the_model(rcdl):
    """The priors are not a tuning knob: a box is (offset from the cell) x (a
    multiplier on its prior), so the prior IS the size. Decoding the same tensors
    with unit priors must collapse every box — this is the contrast that makes
    "18 plausible boxes" mean something. Measured: median area 12 px^2 against
    ~3000, and nothing matching a real vehicle."""
    need(rcdl, "AnchorDetector")
    img = bm.load_bgr("cityscapes.png")
    good, _, _, _ = _yolop_run(rcdl, img)
    bad, _, _, _ = _yolop_run(rcdl, img, anchors=[[(1, 1)] * 3] * 3)

    area = lambda ds: float(np.median([(d.x2 - d.x1) * (d.y2 - d.y1) for d in ds]))
    print(f"\ncorrect priors: {len(good)} boxes, median {area(good):.0f} px^2; "
          f"unit priors: {len(bad)} boxes, median {area(bad):.0f} px^2")
    assert bad, "the objectness gate is what survives, so boxes should still appear"
    assert area(bad) < area(good) / 50
    assert max((_iou(a, b) for a in good for b in bad), default=0.0) < 0.5


def test_the_strides_are_checked_against_the_models_grids(rcdl):
    """The strides are configured but the grids come from the model, and the
    channel count is 18 for all three heads — so nothing about the tensors
    themselves would object if the two lists were paired the wrong way round.
    Every box would land at the wrong position and 4x the wrong size, silently.
    Reversing them must raise instead."""
    need(rcdl, "AnchorDetector")
    engine, cfg = _yolop(rcdl)
    cfg.strides = [32, 16, 8]                      # P5 first: the plausible mistake
    det = rcdl.AnchorDetector(engine._e, cfg, 0)
    img = bm.load_bgr("cityscapes.png")
    drive = rcdl.Segmenter(engine._e, num_classes=2, output_index=3)
    rcdl.segment(drive, img)
    with pytest.raises(Exception):
        det.postprocess(drive.letterbox)


def test_yolop_is_reproducible(rcdl):
    """Every unit on this path is deterministic on its own; only comparing two
    complete runs field by field catches a preprocessing race (see the letterbox
    border in docs/RGA.md)."""
    need(rcdl, "AnchorDetector", "Segmenter")
    img = bm.load_bgr("cityscapes.png")
    d1, dr1, ln1, _ = _yolop_run(rcdl, img)
    d2, dr2, ln2, _ = _yolop_run(rcdl, img)
    assert len(d1) == len(d2)
    for a, b in zip(d1, d2):
        assert (a.x1, a.y1, a.x2, a.y2, a.score, a.class_id) == (
            b.x1, b.y1, b.x2, b.y2, b.score, b.class_id)
    np.testing.assert_array_equal(np.asarray(dr1.labels), np.asarray(dr2.labels))
    np.testing.assert_array_equal(np.asarray(ln1.labels), np.asarray(ln2.labels))


# --------------------------------------------------------------------------- #
# Face recognition — ArcFace R50 identity embeddings                            #
# --------------------------------------------------------------------------- #
ARCFACE_MODEL = "arcface_r50_112_fp16_rk3588.rknn"


def _faces_with_landmarks(rcdl, names=("zidane.jpg", "bus.jpg")):
    """Every face the detector finds, with its five landmarks in source pixels."""
    det = rcdl.Engine(bm.require_model("retinaface_rk3588.rknn")).face_detector()
    out = []
    for name in names:
        img = bm.load_bgr(name)
        for i, f in enumerate(rcdl.detect_faces(det, img)):
            lm = np.array(f.landmarks, np.float32).reshape(5, 2)
            out.append((f"{name}#{i}", img, lm))
    return out


def test_identity_embeddings_separate_different_people(rcdl):
    """The only claim an identity model makes: two vectors of one person score
    high, two of different people score low. This repo carries four faces and
    they are four DIFFERENT people, so this is the negative half — every pair
    must be near zero. Measured: -0.10 to +0.08, against a same-face floor of
    0.98 in the test below."""
    need(rcdl, "FaceRecognizer")
    rec = rcdl.Engine(bm.require_model(ARCFACE_MODEL)).face_recognizer()
    faces = _faces_with_landmarks(rcdl)
    assert len(faces) >= 4, f"expected four faces across the samples, got {len(faces)}"

    vecs = {n: rcdl.embed_face(rec, img, lm) for n, img, lm in faces}
    for n, v in vecs.items():
        assert v.shape == (rec.dim,)
        assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-4), "not unit length"

    names = list(vecs)
    worst = -1.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = float(np.dot(vecs[names[i]], vecs[names[j]]))
            print(f"\n  {names[i]:<13} vs {names[j]:<13} {c:+.3f}")
            worst = max(worst, c)
    assert worst < 0.35, f"two different people scored {worst:.3f} — identities are not separated"


def test_identity_survives_nuisance_but_not_a_missing_alignment(rcdl):
    """Two halves of one contract, and the second is why alignment lives inside
    the recognizer.

    The same face rotated, scaled, dimmed, blurred or JPEG-crushed is the same
    identity by construction, and must stay near 1.0 — measured 0.980-0.999.
    The same face CROPPED TO ITS BOX instead of warped onto the five-point
    template scores 0.493 against its own aligned vector: still positive, so
    nothing looks broken, but half the identity is gone. Nothing in the output
    says the crop was wrong, which is exactly the failure this pins."""
    need(rcdl, "FaceRecognizer")
    import cv2

    rec = rcdl.Engine(bm.require_model(ARCFACE_MODEL)).face_recognizer()
    name, img, lm = _faces_with_landmarks(rcdl, ("zidane.jpg",))[0]
    base = rcdl.embed_face(rec, img, lm)
    h, w = img.shape[:2]

    def warped(m):
        p = np.hstack([lm, np.ones((5, 1), np.float32)])
        return cv2.warpAffine(img, m, (w, h)), (p @ m.T).astype(np.float32)

    worst = 1.0
    cases = [("rot +8", cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.0).astype(np.float32)),
             ("rot -8", cv2.getRotationMatrix2D((w / 2, h / 2), -8, 1.0).astype(np.float32)),
             ("scale 0.85", cv2.getRotationMatrix2D((w / 2, h / 2), 0, 0.85).astype(np.float32))]
    for label, m in cases:
        im, pts = warped(m)
        c = float(np.dot(base, rcdl.embed_face(rec, im, pts)))
        print(f"\n  {label:<11} {c:+.3f}")
        worst = min(worst, c)
    for label, im in (("bright", np.clip(img.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)),
                      ("dark", np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)),
                      ("blur", cv2.GaussianBlur(img, (3, 3), 0)),
                      ("jpeg40", cv2.imdecode(cv2.imencode(
                          ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])[1], 1))):
        c = float(np.dot(base, rcdl.embed_face(rec, im, lm)))
        print(f"  {label:<11} {c:+.3f}")
        worst = min(worst, c)
    assert worst > 0.90, f"a nuisance transform moved the identity to {worst:.3f}"

    # The contrast: same face, box crop instead of the five-point warp.
    x1, y1 = lm[:, 0].min() - 30, lm[:, 1].min() - 40
    x2, y2 = lm[:, 0].max() + 30, lm[:, 1].max() + 40
    box = img[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
    crop = cv2.resize(box, (rec.input_width, rec.input_height))
    v = rec.embed_aligned(np.ascontiguousarray(crop).reshape(-1),
                          rec.input_width, rec.input_height, "bgr888")
    v = v / np.linalg.norm(v)
    misaligned = float(np.dot(base, v))
    print(f"  box crop    {misaligned:+.3f}   (nuisance floor {worst:+.3f})")
    assert misaligned < worst - 0.3, (
        "a box crop scored as well as a real nuisance transform — either the "
        "alignment is not being applied, or this contrast has stopped meaning anything")


def test_face_embeddings_are_reproducible(rcdl):
    need(rcdl, "FaceRecognizer")
    rec = rcdl.Engine(bm.require_model(ARCFACE_MODEL)).face_recognizer()
    name, img, lm = _faces_with_landmarks(rcdl, ("zidane.jpg",))[0]
    np.testing.assert_array_equal(rcdl.embed_face(rec, img, lm),
                                  rcdl.embed_face(rec, img, lm))


def test_face_recognizer_rejects_a_wrongly_sized_aligned_crop(rcdl):
    """embed_aligned takes the model's input size and nothing else: a crop of
    another size would be read as though it were aligned, and the vector would
    look perfectly normal."""
    need(rcdl, "FaceRecognizer")
    rec = rcdl.Engine(bm.require_model(ARCFACE_MODEL)).face_recognizer()
    bad = np.zeros((64, 64, 3), np.uint8)
    with pytest.raises(Exception):
        rec.embed_aligned(np.ascontiguousarray(bad).reshape(-1), 64, 64, "bgr888")


def test_the_float_yolop_runs_and_says_what_int8_costs(rcdl):
    """The fp16 build, and the measurement that decided which build ships.

    It is off every ordinary path here: its input is float32, so engineInputView
    refuses it and no BoundTask can drive it — which is exactly why it fell out
    of the test suite while its numbers went into docs/MODELS.md. Driving it by
    hand is the point of this test, twice over: it pins that the float build
    still loads and runs, and it reproduces the int8-vs-fp16 comparison rather
    than leaving a documented number nobody can check.

    Measured: the two builds agree on 99.7% of DRIVABLE pixels at IoU 0.975, and
    on 99.8% of LANE pixels at IoU 0.762. Same masks, two verdicts — lane lines
    are 1% of the frame, so agreement is dominated by the 99% that is correctly
    not a lane. On a sparse structure, score IoU."""
    need(rcdl, "AnchorDetector")
    import cv2

    fp16 = bm.require_model("yolop_cut_640_fp16_rk3588.rknn")
    i8 = bm.require_model(YOLOP_MODEL)
    img = bm.load_bgr("cityscapes.png")

    # One canvas, both models: any difference is the number format, not the crop.
    canvas, lb, backend = rcdl.letterbox(img, 640, 640, "bgr888", "rgb888")

    def masks(model, feed):
        e = rcdl.Engine(model)
        feed(e)
        e.run()
        return [np.asarray(e.output(i)).reshape(2, 640, 640).argmax(0).astype(np.uint8)
                for i in (3, 4)]

    da8, ll8 = masks(i8, lambda e: e.set_input(0, np.ascontiguousarray(canvas)))
    # float32 STILL IN 0..255 — the (x-mean)/std is folded into the .rknn, and
    # feeding 0..1 here returns a well-formed mask of a picture 255x too dark.
    da16, ll16 = masks(fp16,
                       lambda e: e.set_input(0, np.ascontiguousarray(canvas.astype(np.float32))))

    for name, a, b, lo in (("drivable", da8, da16, 0.90), ("lane", ll8, ll16, 0.60)):
        agree = float((a == b).mean())
        iou = float(np.logical_and(a, b).sum()) / max(1, int(np.logical_or(a, b).sum()))
        print(f"\n  {name}: agreement {100 * agree:.2f}%  IoU {iou:.3f}")
        assert b.any(), f"the float build produced an empty {name} mask"
        assert agree > 0.99
        assert iou > lo, f"{name} IoU {iou:.3f} — the two builds disagree more than measured"

    # The point of the whole comparison: on the sparse head, the two metrics
    # disagree wildly, and only one of them is telling the truth.
    lane_agree = float((ll8 == ll16).mean())
    lane_iou = float(np.logical_and(ll8, ll16).sum()) / max(1, int(np.logical_or(ll8, ll16).sum()))
    assert lane_agree - lane_iou > 0.15, (
        "pixel agreement and IoU no longer diverge on the lane mask — if that is "
        "real the docs' argument for reading IoU needs revisiting")
