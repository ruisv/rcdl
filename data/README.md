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

## audio/ — speech samples for ASR

Four utterances from the **AISHELL-1** test set (16 kHz mono PCM16, 2.7–3.9 s),
with their reference transcripts in `transcripts.tsv` (`file<TAB>text`). Used by
`tests/test_asr_board_py.py`, which expects the Paraformer build to transcribe
all four exactly. Clips 2–4 end on a character that only the CIF tail weight
fires, so they also catch a CIF that loses it. AISHELL-1 is released under the
Apache License 2.0 (OpenSLR SLR33), so the clips ship with the repo.

## kws/ — keyword-spotting reference

`kws_encoder_ref.npz` (`encoder_out`, 92 × 320): the float KWS encoder's output
for `audio/aishell1_test_1.wav`, streamed chunk by chunk as on the board.
`tests/test_kws_board_py.py` checks the NPU build against it; the wake words
themselves are synthesised on the board with `TextToSpeech`.

## vad/ — VAD reference

`silero_ref.npz` (`probs`): the float Silero VAD v6.2.3 ONNX's speech
probability for each 512-sample chunk of the signal `tests/test_vad_board_py.py`
builds from `audio/` (0.5 s silence, clip 1, 1.2 s, clip 3, 1.2 s). The NPU build
is checked against it.

## tts/ — vocoder inputs and references

Two Matcha-TTS (zh-en) mel spectrograms, `matcha_mel_00.npy` (80 × 136, one
chunk) and `matcha_mel_36.npy` (80 × 245, two chunks), with the float Vocos
output for each (`vocos_float_*.wav`, 16 kHz PCM16) and the sentences in
`texts.tsv`. Used by `tests/test_vocoder_board_py.py`, which checks the NPU
Vocos build against the float waveforms and runs the result back through
Paraformer. The mels are the acoustic model's output on sherpa-onnx's own front
end (punctuation tokens, sentence split, noise scale 1.0), so they are what the
vocoder sees in deployment.

`frontend_ref.json` holds 100 sentences with the token ids sherpa-onnx's own
front end produces for them (from its debug log); `tests/test_tts_frontend.py`
requires `MatchaFrontend` to reproduce every one. `matcha_ref_00.npz` and
`matcha_ref_36.npz` hold, per sentence, the token ids, the durations and the
float acoustic model's mel with the decoder noise off, which
`tests/test_tts_board_py.py` compares the NPU build against.

