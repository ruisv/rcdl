# data/

Small assets the board tests and examples use, committed so results are
reproducible without depending on anything board-specific. Both roots are
env-overridable: `RCDL_IMAGES`, and `RCDL_MODELS` for the (gitignored) `models/`
directory alongside this one.

## images/

| file | used by | what it is |
|------|---------|------------|
| `bus.jpg` | detection, instance seg, pose, tracking | Ultralytics COCO sample — 1 bus + 4 people, the pinned expectation in `tests/test_detection_board_py.py` |
| `zidane.jpg` | pose, face | two large clear faces and two people, so keypoints and landmarks are easy to judge by eye |
| `ocr.jpg` | OCR | a Chinese product label, 16 text lines including one vertical and one rotated |
| `obb.jpg` | oriented boxes | DOTA aerial scene — planes and a parking row of vehicles |
| `cityscapes.png` | semantic segmentation | the PP-LiteSeg sample street scene (2048×1024) |
| `bird.jpg` | classification | ImageNet sample |
| `space_shuttle_224.jpg` | classification | ImageNet sample, already 224x224 — the pinned expectation is class **812**, "space shuttle" |

**`obb.jpg` is a progressive JPEG**, which the VPU's JPEG core cannot decode
(it does baseline and extended-sequential Huffman only). Read that one with
OpenCV; `rcdl::JpegDecoder` will decline it and say so via `lastError()`.

## OCR character dictionary

`ppocr_keys_v1_6625.txt` — `blank` + PaddleOCR's `ppocr_keys_v1.txt` (6623
entries) + a trailing space = **6625 classes**, which is exactly the channel
count of the `ppocrv4_rec` recognition model. That equality is not incidental:
`TextRecognizer` uses the dictionary size to work out which axis of a
`[1,T,C]` / `[1,C,T]` output is the class axis, so a mismatched dictionary is
detected rather than silently producing garbage text.

Other PP-OCR versions use different tables (v5 is 18385 classes, v6 is 18710);
add the matching one if you deploy those models.
