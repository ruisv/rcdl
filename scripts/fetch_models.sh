#!/usr/bin/env bash
# Populate models/ with the .rknn files the examples, tests and benchmarks use.
#
# Sources, in order: an rknn_model_zoo checkout (RCDL_SRC_ZOO) for Rockchip's
# prebuilt models, then the conversion host (RCDL_CONVERT_HOST +
# RCDL_CONVERT_MODELS) for models converted by the rcdl model-zoo recipes.
# Paths come from scripts/local.env (gitignored; see local.env.example).
# Anything left unset is reported as MISSING rather than silently skipped.
#
#   scripts/fetch_models.sh            # stage everything known
#   scripts/fetch_models.sh --list     # show the registry
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
[ -f "${PROJ}/scripts/local.env" ] && source "${PROJ}/scripts/local.env"
DEST="${PROJ}/models"
mkdir -p "${DEST}"

# name | source kind | relative path at the source
#   zoo     : ${RCDL_SRC_ZOO}/<path>          (rknn_model_zoo checkout; run its
#                                              download_model.sh first)
#   convert : ${RCDL_CONVERT_HOST}:${RCDL_CONVERT_MODELS}/<path>
# Each model's INPUT ORDER (rgb/bgr) and head family are recorded here so the
# examples and tests do not have to guess; see docs/MODELS.md for the table.
REGISTRY=(
  "yolov8n_rk3588.rknn|convert|yolov8n_rk3588.rknn"
  "yolo11n_rk3588.rknn|convert|yolo11n_rk3588.rknn"
  # YOLO26n: NMS-free head, and — the part that matters to the decoder — NO DFL,
  # so the box branch is 4 channels instead of 64 and postprocessing is half the
  # cost. Exported from the one2one branch (see docs/MODELS.md).
  "yolo26n_rk3588.rknn|convert|yolo26n_rk3588.rknn"
  "yolo26n-seg_rk3588.rknn|convert|yolo26n-seg_rk3588.rknn"
  # Pose needs the DECODER told which generation it is: YOLO26 dropped the
  # doubling in the keypoint offset (kpt_decode="cell_relative_whole").
  "yolo26n-pose_rk3588.rknn|convert|yolo26n-pose_rk3588.rknn"
  # OBB: YOLO26 regresses the angle in RADIANS, so decode with angle_bias=0 and
  # angle_scale=1 (the v8 default is (sigmoid-0.25)*pi). Classification: the
  # softmax is IN the graph, so pass apply_softmax=False.
  "yolo26n-obb_rk3588.rknn|convert|yolo26n-obb_rk3588.rknn"
  "yolo26n-cls_rk3588.rknn|convert|yolo26n-cls_rk3588.rknn"
  "yolov8n-pose_rk3588.rknn|convert|yolov8n-pose_rk3588.rknn"
  "yolov8n-seg_rk3588.rknn|convert|yolov8n-seg_rk3588.rknn"
  "yolov8n-obb_rk3588.rknn|convert|yolov8n-obb_rk3588.rknn"
  "ppocrv4_det_rk3588.rknn|convert|ppocrv4_det_rk3588.rknn"
  "ppocrv4_rec_rk3588.rknn|convert|ppocrv4_rec_rk3588.rknn"
  # Text-line direction (0/180). BGR in, 48x192, softmax in the graph. It goes
  # between the detector and the recogniser: a CTC model fed an upside-down line
  # reads out confident nonsense rather than failing. See docs/MODELS.md.
  "ppocr_cls_rk3588.rknn|convert|ppocr_cls_rk3588.rknn"
  # PP-OCRv5 mobile detection. Measures identical to the v4 detector on the
  # sample page (same boxes, same text); kept as the newer alternative and
  # pinned by a parity test.
  "ppocrv5_det_rk3588.rknn|convert|ppocrv5_det_rk3588.rknn"
  # PP-OCRv5 server recognition, exported to emit LOGITS: the 18385-way softmax
  # is the one op that does not survive this runtime's f16 path (it comes back
  # summing to 0.64-1.00 per step with no peak, where the simulator peaks at
  # 1.000), and CTC only needs the argmax. Decode with apply_softmax=True and
  # feed it RAW 0..255 — input_scale=1, input_shift=0. Reads the sample page at
  # mean 0.928 against the v4 build's 0.661. See docs/MODELS.md.
  "ppocrv5_server_rec_logits_rk3588.rknn|convert|ppocrv5_server_rec_logits_rk3588.rknn"
  "ppseg_rk3588.rknn|convert|ppseg_rk3588.rknn"
  "retinaface_rk3588.rknn|convert|retinaface_rk3588.rknn"
  "resnet18_rk3588.rknn|convert|resnet18_rk3588.rknn"
  # Monocular depth. The 308x308 build is the one the tests use: it is 2.9x
  # faster than the native 518x518 build AND closer to the fp32 reference
  # (see docs/MODELS.md). The 518 build is kept for the native-resolution case.
  "depth_anything_v2_vits_308_rk3588.rknn|convert|depth_anything_v2_vits_308_rk3588.rknn"
  "depth_anything_v2_vits_rk3588.rknn|convert|depth_anything_v2_vits_rk3588.rknn"
  # Person ReID appearance vectors — what ImageEmbedder/EmbeddingBank and the
  # ReID side of ByteTrack consume. 512-d, crops squashed to 128x256 (NOT
  # letterboxed); see docs/MODELS.md.
  "osnet_x0_25_msmt17_rk3588.rknn|convert|osnet_x0_25_msmt17_rk3588.rknn"
  # XFeat sparse features, 640x480. Its input is NOT image bytes: the graph
  # starts after the reference's InstanceNorm, so open the Engine with
  # float_inputs=[0] (FeatureExtractor refuses otherwise). int8 measured
  # equivalent to the float ONNX on a known-warp match test; see docs/MODELS.md.
  "xfeat_640x480_i8_rk3588.rknn|convert|xfeat_640x480_i8_rk3588.rknn"
  # x4 super-resolution, 128x128 tile. The fp16 build is the default: it
  # reproduces the float model to 63 dB where int8 manages 31 and over-sharpens.
  # int8 is ~1.6x faster and stays for when that trade is worth it.
  # Both take 0..255 (bytes for int8, FLOATS still in 0..255 for fp16) — the
  # /255 is inside the model. See docs/MODELS.md.
  "realesr_general_x4v3_128_fp16_rk3588.rknn|convert|realesr_general_x4v3_128_fp16_rk3588.rknn"
  "realesr_general_x4v3_128_i8_rk3588.rknn|convert|realesr_general_x4v3_128_i8_rk3588.rknn"
  # Dense optical flow, 512x384, two frames in. Built with its GridSample nodes
  # declared as a CUSTOM OPERATOR (`cstGridSample`) — librknnrt 2.3.2 implements
  # that op nowhere, and a build without the declaration segfaults inside
  # rknn_init. RCDL registers the CPU kernel itself (backend/custom_ops.h).
  "neuflow_v2_512x384_fp16_rk3588.rknn|convert|neuflow_v2_512x384_fp16_rk3588.rknn"
  # Promptable segmentation (EdgeSAM): encode once at 1024x1024, then one
  # decoder pass per prompt. BOTH are float on measurement — the int8 encoder
  # keeps large shapes and loses small ones entirely, so it stays recipe-only
  # rather than shipping as a trap. See docs/MODELS.md.
  "edge_sam_3x_encoder_fp16_rk3588.rknn|convert|edge_sam_3x_encoder_fp16_rk3588.rknn"
  "edge_sam_3x_decoder_fp16_rk3588.rknn|convert|edge_sam_3x_decoder_fp16_rk3588.rknn"
  # Whole-body pose (RTMW): 133 keypoints for ONE person's box, top-down, so a
  # detector runs first and this runs once per person. Float on measurement —
  # the int8 build keeps easy crops and drops 111 of 133 joints on a hard one.
  # Input is float32 in 0..255 (ImageNet mean/std folded in); the head is SimCC,
  # two bins per input pixel. See docs/MODELS.md.
  "rtmw_s_133_256x192_fp16_rk3588.rknn|convert|rtmw_s_133_256x192_fp16_rk3588.rknn"
  # Open-vocabulary detection (YOLOE-11s). The vocabulary is a CONVERSION-time
  # parameter: the CLIP text encoder runs on the convert host and its output is
  # folded into the classification conv, so what lands here is an ordinary
  # 9-output LTRB head with one class channel per word — decoded by the same
  # code as every other YOLO. The `.labels.txt` beside each model is the only
  # runtime state (rcdl::LabelMap); load them together.
  "yoloe_11s_coco80_rk3588.rknn|convert|yoloe_11s_coco80_rk3588.rknn"
  "yoloe_11s_coco80_rk3588.labels.txt|convert|yoloe_11s_coco80_rk3588.labels.txt"
  # The same checkpoint with six words COCO has no class for at all. This is the
  # build that shows the vocabulary is load-bearing rather than decorative.
  "yoloe_11s_streetwear_rk3588.rknn|convert|yoloe_11s_streetwear_rk3588.rknn"
  "yoloe_11s_streetwear_rk3588.labels.txt|convert|yoloe_11s_streetwear_rk3588.labels.txt"
  # Panoptic driving (YOLOP): one inference, three heads — an anchor-BASED
  # detector plus drivable-area and lane-line masks. "_cut" because the
  # published export bakes the anchor decode into the graph with ScatterND;
  # that compiles cleanly into a tensor whose objectness and class columns are
  # never written. int8 is the build the tests and benchmarks use, because the
  # fp16 one takes float32 input and so cannot be letterboxed straight into the
  # NPU tensor. The trade it makes: every box and 0.905 drivable IoU kept, but a
  # third of the LANE mask lost (IoU 0.610 against fp16's 0.969). Take fp16 when
  # lane geometry is the point and a CPU input pass is affordable. See
  # docs/MODELS.md.
  "yolop_cut_640_fp16_rk3588.rknn|convert|yolop_cut_640_fp16_rk3588.rknn"
  "yolop_cut_640_i8_rk3588.rknn|convert|yolop_cut_640_i8_rk3588.rknn"
  # Face recognition (ArcFace R50). FLOAT for a DATA reason rather than an
  # accuracy one: int8 would need a calibration set of ALIGNED 112x112 crops —
  # calibrating this network on centre crops gives a different model whose
  # published accuracy does not apply — and this repo carries four faces, all
  # different people. fp16 needs no calibration set and reproduces the fp32 ONNX
  # to cosine 1.00000 on an identical crop. Input is float32 still in 0..255.
  "arcface_r50_112_fp16_rk3588.rknn|convert|arcface_r50_112_fp16_rk3588.rknn"
  # Open-vocabulary INSTANCE SEGMENTATION, from the same YOLOE checkpoint as the
  # detector above with the mask branch exported too: 13 outputs, which is the
  # yolov8n-seg layout, so InstanceSegmenter reads it with no changes at all.
  "yoloe_11s_coco80_seg_rk3588.rknn|convert|yoloe_11s_coco80_seg_rk3588.rknn"
  "yoloe_11s_coco80_seg_rk3588.labels.txt|convert|yoloe_11s_coco80_seg_rk3588.labels.txt"
  # YOLO semantic segmentation, the same 19 Cityscapes classes PP-LiteSeg has —
  # which is what makes the two checkable against each other. Exported as LOGITS
  # at [1,19,80,80]: ultralytics' ONNX export bakes the argmax in and returns a
  # [1,H,W] map, which cannot be scored and would need its own decode path.
  # Calibrated on dashcam frames rather than COCO — see docs/MODELS.md.
  "yolo26n_sem_640_i8_rk3588.rknn|convert|yolo26n_sem_640_i8_rk3588.rknn"
  "yolo26n_sem_640_fp16_rk3588.rknn|convert|yolo26n_sem_640_fp16_rk3588.rknn"
  # PP-OCRv6 medium recognition, 18710 classes, same logits export as v5 — the
  # softmax is out of the graph and applied on the CPU (apply_softmax=True, raw
  # 0..255 in). The most confident of the three recognisers here (0.930 on the
  # sample page against v5's 0.928 and v4's 0.661), and it agrees with v5 on the
  # one character v4 reads differently. Needs data/ppocr_keys_v6_18710.txt.
  "ppocrv6_medium_rec_logits_rk3588.rknn|convert|ppocrv6_medium_rec_logits_rk3588.rknn"
  # PP-OCRv6 medium detection, 480x480. Converted from the ONNX PaddleOCR
  # PUBLISHES for v6, not from its Paddle inference model: paddle2onnx aborts on
  # the v6 PIR export, and going through it is unnecessary because upstream ships
  # ONNX (fix the dynamic shapes with onnxslim and that is the whole export).
  # Same int8 recipe and calibration set as the v4 and v5 detectors so the three
  # are comparable; decode it with PP-OCRv6's own DB thresholds (bin 0.2,
  # box 0.45, unclip 1.4) rather than the library defaults — see docs/MODELS.md.
  "ppocrv6_medium_det_rk3588.rknn|convert|ppocrv6_medium_det_rk3588.rknn"
  # SigLIP base/16 — the image half of an image-text pair. The TEXT tower runs on
  # the conversion host once per label set, exactly as YOLOE's does, so what
  # reaches the board is an ordinary image encoder and zero-shot classification
  # is a dot product against the table below. FLOAT: int8 destroys this network
  # (every label lands within noise of every other), which is what a ViT does
  # under PTQ. See docs/MODELS.md.
  "siglip_b16_224_fp16_rk3588.rknn|convert|siglip_b16_224_fp16_rk3588.rknn"
  "siglip_b16_224_text_coco80.npy|convert|siglip_b16_224_text_coco80.npy"
  "siglip_b16_224_labels.txt|convert|siglip_b16_224_labels.txt"
)

if [ "${1:-}" = "--list" ]; then
  printf '%s\n' "${REGISTRY[@]}" | column -t -s'|'
  exit 0
fi

missing=0
for entry in "${REGISTRY[@]}"; do
  IFS='|' read -r name kind rel <<<"${entry}"
  out="${DEST}/${name}"
  if [ -s "${out}" ]; then echo "   ok  ${name}"; continue; fi
  case "${kind}" in
    zoo)
      src="${RCDL_SRC_ZOO:-}/${rel}"
      if [ -n "${RCDL_SRC_ZOO:-}" ] && [ -s "${src}" ]; then
        cp "${src}" "${out}" && echo "  got  ${name}  (zoo)"
      else
        echo "MISSING ${name}  — set RCDL_SRC_ZOO and download it there (${rel})"; missing=$((missing+1))
      fi ;;
    convert)
      if [ -n "${RCDL_CONVERT_HOST:-}" ] && [ -n "${RCDL_CONVERT_MODELS:-}" ]; then
        scp -q "${RCDL_CONVERT_HOST}:${RCDL_CONVERT_MODELS}/${rel}" "${out}" \
          && echo "  got  ${name}  (convert host)" \
          || { echo "MISSING ${name}  — not on the convert host (${rel})"; missing=$((missing+1)); }
      else
        echo "MISSING ${name}  — set RCDL_CONVERT_HOST / RCDL_CONVERT_MODELS"; missing=$((missing+1))
      fi ;;
  esac
done
echo ">> models/: $(ls -1 "${DEST}"/*.rknn 2>/dev/null | wc -l | tr -d ' ') staged, ${missing} missing"
