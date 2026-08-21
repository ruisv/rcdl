"""Shared on-board model/image resolution for the end-to-end tests.

Every `.rknn` lives in the repo-local `models/` directory (gitignored, populated
by `scripts/fetch_models.sh`) and every sample image in `data/images/`, so the
board tests carry no absolute paths and no board-specific asset locations. Both
roots are env-overridable: `RCDL_MODELS`, `RCDL_IMAGES`.

A test that needs a model it cannot find SKIPS — the suite must stay green on a
machine that has only some of the registry staged.
"""

import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.environ.get("RCDL_MODELS", os.path.join(_REPO, "models"))
IMAGES = os.environ.get("RCDL_IMAGES", os.path.join(_REPO, "data", "images"))


def find_model(*names):
    """First of `names` that exists under MODELS, else None.

    Several names are accepted because the same network reaches the board under
    slightly different file names depending on how it was converted.
    """
    for n in names:
        if os.path.isabs(n) and os.path.isfile(n):
            return n
        p = os.path.join(MODELS, n)
        if os.path.isfile(p):
            return p
    return None


def require_model(*names):
    import pytest
    p = find_model(*names)
    if p is None:
        pytest.skip(f"model not staged in {MODELS}: {' | '.join(names)} "
                    f"(run scripts/fetch_models.sh)")
    return p


def find_image(name):
    if os.path.isabs(name):
        return name
    return os.path.join(IMAGES, name)


def require_image(name):
    import pytest
    p = find_image(name)
    if not os.path.isfile(p):
        pytest.skip(f"sample image missing: {p}")
    return p


def load_bgr(name):
    """Read a sample image as an HxWx3 uint8 BGR array (needs OpenCV)."""
    import pytest
    cv2 = pytest.importorskip("cv2", reason="OpenCV needed to decode the sample images")
    img = cv2.imread(require_image(name), cv2.IMREAD_COLOR)
    if img is None:
        pytest.skip(f"could not decode {name}")
    return img


# COCO-80 in YOLO export order — mirrors rcdl::cocoClassNames().
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
