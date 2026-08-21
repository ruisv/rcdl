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
