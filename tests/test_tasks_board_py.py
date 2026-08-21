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
    import os
    img_path = os.environ.get("RCDL_SHUTTLE_IMAGE", "")
    if not img_path or not os.path.isfile(img_path):
        pytest.skip("set RCDL_SHUTTLE_IMAGE to the ImageNet space-shuttle sample")
    cv2 = pytest.importorskip("cv2")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
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
