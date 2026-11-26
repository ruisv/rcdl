#!/usr/bin/env bash
# Populate models/ with the .rknn files the examples, tests and benchmarks use.
#
# Published builds come from Hugging Face (the `hf` rows below) and are checked
# against the SHA-256 recorded here. HF_ENDPOINT points the download at a mirror.
# The rest are not published, and model_zoo/README.md says why for each. They
# come from a conversion host (RCDL_CONVERT_HOST + RCDL_CONVERT_MODELS) or an
# rknn_model_zoo checkout (RCDL_SRC_ZOO), configured in scripts/local.env
# (gitignored; see local.env.example). A published file the download cannot
# reach also falls back to the conversion host when one is configured. Anything
# left unavailable is reported as MISSING rather than silently skipped.
#
#   scripts/fetch_models.sh                       # stage everything available
#   scripts/fetch_models.sh yolov8n_rk3588.rknn   # only the files named
#   scripts/fetch_models.sh --list                # show the registry
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
[ -f "${PROJ}/scripts/local.env" ] && source "${PROJ}/scripts/local.env"
DEST="${RCDL_MODELS:-${PROJ}/models}"   # the same override the tests and examples honour
mkdir -p "${DEST}"

# name | source kind | where | sha256 (hf rows)
#   hf      : ${HF_ENDPOINT:-https://huggingface.co}/<repo>/resolve/main/<name>
#             (model_zoo/publish_hf.py uploads exactly these rows)
#   zoo     : ${RCDL_SRC_ZOO}/<path>          (rknn_model_zoo checkout; run its
#                                              download_model.sh first)
#   convert : ${RCDL_CONVERT_HOST}:${RCDL_CONVERT_MODELS}/<path>
# Each model's INPUT ORDER (rgb/bgr) and head family are recorded here so the
# examples and tests do not have to guess; see docs/MODELS.md for the table.
REGISTRY=(
  "yolov8n_rk3588.rknn|hf|ruisv/rcdl-yolo|ab1ce6a515812e072e4c486e25fa4277285a78c0eae98b3555b79e92ce530d7e"
  "yolo11n_rk3588.rknn|hf|ruisv/rcdl-yolo|2c85742eb4970c7b41ca43d5b0014d847b89540ff768a1f537deb738e1d11031"
  # YOLO26n: NMS-free head, and — the part that matters to the decoder — NO DFL,
  # so the box branch is 4 channels instead of 64 and postprocessing is half the
  # cost. Exported from the one2one branch (see docs/MODELS.md).
  "yolo26n_rk3588.rknn|hf|ruisv/rcdl-yolo|025d7827b1fe95d9125debc21ababb05d667562a1eecfb5d4dc3567886e3ae1b"
  "yolo26n-seg_rk3588.rknn|hf|ruisv/rcdl-yolo|1567e547f9522347192b3f63b94bdd690da5903181af081677859c2a9b7f83e0"
  # Pose needs the DECODER told which generation it is: YOLO26 dropped the
  # doubling in the keypoint offset (kpt_decode="cell_relative_whole").
  "yolo26n-pose_rk3588.rknn|hf|ruisv/rcdl-yolo|07259facc63d9a50b187ede0b2ed3a8588260def503fdc013e9731a7c2db74dc"
  # OBB: YOLO26 regresses the angle in RADIANS, so decode with angle_bias=0 and
  # angle_scale=1 (the v8 default is (sigmoid-0.25)*pi). Classification: the
  # softmax is IN the graph, so pass apply_softmax=False.
  "yolo26n-obb_rk3588.rknn|hf|ruisv/rcdl-yolo|020673a13a5ff7865cc643096ea9a7b49f1bc8ff9b778ea7c7db199fe181dcd2"
  "yolo26n-cls_rk3588.rknn|hf|ruisv/rcdl-yolo|371709695f2f1dd6fccf0261054ca7c29a6a5977f8a4c869b2fde9f1fb89bc85"
  "yolov8n-pose_rk3588.rknn|hf|ruisv/rcdl-yolo|fbc986f3762b9a62961aec95a7672534fb70aeca3edfce19140f16e5e7e267d7"
  "yolov8n-seg_rk3588.rknn|hf|ruisv/rcdl-yolo|56f08877116bc9b6446cb0e11545d9b9a4e9d6c1dbc49a026ed2eeea232ba243"
  "yolov8n-obb_rk3588.rknn|hf|ruisv/rcdl-yolo|9a39bbc383dbd798b84248a51ca608c9817d9444092ad0fd91cd09e4dbab6a52"
  "ppocrv4_det_rk3588.rknn|hf|ruisv/rcdl-ppocr|339fa8840518b89f67b584b6de27c4d2983e6034b4b00d4ff67ec546a70d5dbc"
  "ppocrv4_rec_rk3588.rknn|hf|ruisv/rcdl-ppocr|842323890dcd1bb871b2b66a0e8ade617cc58832430b6fe0fbfc45dbf82532f5"
  # Text-line direction (0/180). BGR in, 48x192, softmax in the graph. It goes
  # between the detector and the recogniser: a CTC model fed an upside-down line
  # reads out confident nonsense rather than failing. See docs/MODELS.md.
  "ppocr_cls_rk3588.rknn|hf|ruisv/rcdl-ppocr|77f7cc59cd9705996a5d3012f320097255f22110627d6aa898df20966fd9c741"
  # PP-OCRv5 mobile detection. Measures identical to the v4 detector on the
  # sample page (same boxes, same text); kept as the newer alternative and
  # pinned by a parity test.
  "ppocrv5_det_rk3588.rknn|hf|ruisv/rcdl-ppocr|96ca53204f6e5d1b991d9783861fee4276fb3df2d1e5ec14c7077a04b5cb6637"
  # PP-OCRv5 server recognition, exported to emit LOGITS: the 18385-way softmax
  # is the one op that does not survive this runtime's f16 path (it comes back
  # summing to 0.64-1.00 per step with no peak, where the simulator peaks at
  # 1.000), and CTC only needs the argmax. Decode with apply_softmax=True and
  # feed it RAW 0..255 — input_scale=1, input_shift=0. Reads the sample page at
  # mean 0.928 against the v4 build's 0.661. See docs/MODELS.md.
  "ppocrv5_server_rec_logits_rk3588.rknn|hf|ruisv/rcdl-ppocr|d8c20d0a810dbbc2bcb3607ff632bb37ab289bb025ed0117105d71a0cf6c3791"
  "ppseg_rk3588.rknn|convert|ppseg_rk3588.rknn"
  "retinaface_rk3588.rknn|convert|retinaface_rk3588.rknn"
  "resnet18_rk3588.rknn|convert|resnet18_rk3588.rknn"
  # Monocular depth. The 308x308 build is the one the tests use: it is 2.9x
  # faster than the native 518x518 build AND closer to the fp32 reference
  # (see docs/MODELS.md). The 518 build is kept for the native-resolution case.
  "depth_anything_v2_vits_308_rk3588.rknn|hf|ruisv/rcdl-depth-anything-v2-small|57827e44f48fbfd7bffea4dad86320888397cf9e4a5ebb9b322df56598bc0f65"
  "depth_anything_v2_vits_rk3588.rknn|hf|ruisv/rcdl-depth-anything-v2-small|6a9609d8eaed735efb1a39c820fc7009c469eac831d6eb80aa2bd98f4f2aef55"
  # Person ReID appearance vectors — what ImageEmbedder/EmbeddingBank and the
  # ReID side of ByteTrack consume. 512-d, crops squashed to 128x256 (NOT
  # letterboxed); see docs/MODELS.md.
  "osnet_x0_25_msmt17_rk3588.rknn|convert|osnet_x0_25_msmt17_rk3588.rknn"
  # XFeat sparse features, 640x480. Its input is NOT image bytes: the graph
  # starts after the reference's InstanceNorm, so open the Engine with
  # float_inputs=[0] (FeatureExtractor refuses otherwise). int8 measured
  # equivalent to the float ONNX on a known-warp match test; see docs/MODELS.md.
  "xfeat_640x480_i8_rk3588.rknn|hf|ruisv/rcdl-xfeat|f8fd597764f88df158f35f57b5cc7ad3bdc5a288a32388f9aa47360e0025b884"
  # x4 super-resolution, 128x128 tile. The fp16 build is the default: it
  # reproduces the float model to 63 dB where int8 manages 31 and over-sharpens.
  # int8 is ~1.6x faster and stays for when that trade is worth it.
  # Both take 0..255 (bytes for int8, FLOATS still in 0..255 for fp16) — the
  # /255 is inside the model. See docs/MODELS.md.
  "realesr_general_x4v3_128_fp16_rk3588.rknn|hf|ruisv/rcdl-superres|c722ca1019031fb1c5155204f4b666d49116869a4397951cfa7ba0e9c381330e"
  "realesr_general_x4v3_128_i8_rk3588.rknn|hf|ruisv/rcdl-superres|842f0a1b9fcec63f16167eb587ad44acc189488454e252156eed8d53d96e5caa"
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
  "ppocrv6_medium_rec_logits_rk3588.rknn|hf|ruisv/rcdl-ppocr|c0388378c2979b605d1be7df7e0259f687edd18b22850517e99715eb143910e8"
  # PP-OCRv6 medium detection, 480x480. Converted from the ONNX PaddleOCR
  # PUBLISHES for v6, not from its Paddle inference model: paddle2onnx aborts on
  # the v6 PIR export, and going through it is unnecessary because upstream ships
  # ONNX (fix the dynamic shapes with onnxslim and that is the whole export).
  # Same int8 recipe and calibration set as the v4 and v5 detectors so the three
  # are comparable; decode it with PP-OCRv6's own DB thresholds (bin 0.2,
  # box 0.45, unclip 1.4) rather than the library defaults — see docs/MODELS.md.
  "ppocrv6_medium_det_rk3588.rknn|hf|ruisv/rcdl-ppocr|e7d53a8e8deddcc8c829cd95f0fa139bb783824b1d25e004d5577781d7fda112"
  # SigLIP base/16 — the image half of an image-text pair. The TEXT tower runs on
  # the conversion host once per label set, exactly as YOLOE's does, so what
  # reaches the board is an ordinary image encoder and zero-shot classification
  # is a dot product against the table below. FLOAT: int8 destroys this network
  # (every label lands within noise of every other), which is what a ViT does
  # under PTQ. See docs/MODELS.md.
  "siglip_b16_224_fp16_rk3588.rknn|hf|ruisv/rcdl-siglip-b16|65440e89b95db838e124ebb10005519bd77b4c2ba7727f7e72ba9056ae67022a"
  "siglip_b16_224_text_coco80.npy|hf|ruisv/rcdl-siglip-b16|0cf7d3aeed35708dd0140f824ac6432ce51dd0a622daadd6eff787ed10bf48e8"
  "siglip_b16_224_labels.txt|hf|ruisv/rcdl-siglip-b16|bd17f1ee35d5f3c862a4894605855abbb9dda4b0621fdb0ac4c2c8c7bb7e730a"
)

if [ "${1:-}" = "--list" ]; then
  printf '%s\n' "${REGISTRY[@]}" | cut -d'|' -f1-3 | column -t -s'|'
  exit 0
fi

# Positional arguments narrow the run to the files named. (A plain string, not
# an associative array: macOS still ships bash 3.2.)
WANT=" $* "
for a in "$@"; do
  printf '%s\n' "${REGISTRY[@]}" | cut -d'|' -f1 | grep -qxF -- "${a}" \
    || { echo "not in the registry: ${a} (see --list)" >&2; exit 2; }
done

sha256_of() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# from_convert_host <name> <rel> — the maintainers' conversion host, if configured.
from_convert_host() {
  [ -n "${RCDL_CONVERT_HOST:-}" ] && [ -n "${RCDL_CONVERT_MODELS:-}" ] || return 1
  scp -q "${RCDL_CONVERT_HOST}:${RCDL_CONVERT_MODELS}/$2" "${DEST}/$1"
}

HF="${HF_ENDPOINT:-https://huggingface.co}"
missing=0
for entry in "${REGISTRY[@]}"; do
  IFS='|' read -r name kind rel sha <<<"${entry}"
  if [ "$#" -gt 0 ] && [[ "${WANT}" != *" ${name} "* ]]; then continue; fi
  out="${DEST}/${name}"
  if [ -s "${out}" ]; then
    if [ "${kind}" = hf ] && [ "$(sha256_of "${out}")" != "${sha}" ]; then
      echo "STALE  ${name}  — differs from the published build; delete it to refetch"
    else
      echo "   ok  ${name}"
    fi
    continue
  fi
  case "${kind}" in
    hf)
      if curl -fsSL --retry 3 -o "${out}.part" "${HF}/${rel}/resolve/main/${name}" \
          && [ "$(sha256_of "${out}.part")" = "${sha}" ]; then
        mv -f "${out}.part" "${out}" && echo "  got  ${name}  (${rel})"
      elif rm -f "${out}.part" && from_convert_host "${name}" "${name}"; then
        echo "  got  ${name}  (convert host; the download failed or did not match its SHA-256)"
      else
        echo "MISSING ${name}  — download from ${HF}/${rel} failed or did not match its SHA-256"
        missing=$((missing+1))
      fi ;;
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
        echo "MISSING ${name}  — not published (model_zoo/README.md says why); convert it, or set RCDL_CONVERT_HOST"
        missing=$((missing+1))
      fi ;;
  esac
done
echo ">> models/: $(ls -1 "${DEST}"/*.rknn 2>/dev/null | wc -l | tr -d ' ') staged, ${missing} missing"
