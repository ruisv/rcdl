"""RCDL — RKNPU Computational Deep Learning (Python wrapper).

Thin numpy-friendly layer over the compiled ``rcdl_py`` extension. The C++ core
exchanges raw bytes and float32 buffers; this wrapper keeps the Python API
small and numpy-shaped.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

import rcdl_py
from rcdl_py import (
    AsyncVideoDetectionPipeline,
    Classifier,
    ClsConfig,
    ClsResult,
    CropBox,
    DepthConfig,
    DepthEstimator,
    DepthMap,
    Detection,
    DetectionPipeline,
    DmaBuf,
    DmaHeap,
    EmbedConfig,
    EmbedMatch,
    EmbeddingBank,
    FaceConfig,
    FaceDetection,
    FaceDetector,
    FaceHeadLayout,
    FaceRecognizer,
    FeatureExtractor,
    FeatureSet,
    ImageEmbedder,
    InstanceMask,
    InstanceSegConfig,
    InstanceSegmenter,
    Keypoint,
    NpuCore,
    ObbConfig,
    OpticalFlowEstimator,
    XfeatConfig,
    ObbDetection,
    ObbDetector,
    OcrDetConfig,
    OcrRecConfig,
    PoseConfig,
    PoseDetection,
    PoseEstimator,
    BodyPart,
    CropRect,
    PromptMask,
    PromptableSegmenter,
    WholeBodyEstimator,
    RotatedBox,
    SegMask,
    SuperResConfig,
    SuperResolver,
    Segmenter,
    TextAngleClassifier,
    TextBox,
    TextDetector,
    TextLine,
    TextOrientation,
    TextRecognizer,
    center_crop_box,
    class_count_from_shape,
    class_label,
    coco_class_name,
    coco_class_names,
    coco_keypoint_name,
    coco_keypoint_names,
    coco_skeleton,
    compute_letterbox,
    ctc_greedy_decode,
    decode,
    decode_classification,
    decode_depth,
    decode_embedding,
    arcface_template,
    decode_faces,
    decode_flow,
    flow_colorize,
    flow_endpoint_error,
    flow_preprocess,
    face_align_transform,
    decode_xfeat,
    match_features,
    xfeat_preprocess,
    similarity_transform,
    decode_instance_seg,
    decode_obb,
    body_part,
    body_part_name,
    body_part_range,
    crop_geometry,
    decode_pose,
    decode_simcc,
    encode_box_prompt,
    encode_point_prompt,
    mask_from_logits,
    decode_seg,
    decode_text_boxes,
    decode_text_orientation,
    ocr_line_fit_width,
    decode_yolo_ltrb,
    yolo_head_classes,
    decode_yolov5_anchor,
    Anchor,
    AnchorDetectConfig,
    AnchorDetector,
    LabelMap,
    depth_colorize,
    depth_resize,
    depth_to_gray8,
    depth_to_source,
    dota_class_name,
    dota_class_names,
    embedding_dim_from_shape,
    generate_priors,
    load_char_dict,
    load_class_labels,
    looks_like_probabilities,
    min_area_quad,
    rotated_iou,
    rotated_nms,
    plan_tiles,
    seg_colorize,
    seg_resize,
    seg_to_source,
    sort_text_boxes,
    tile_weight,
    unclip_quad,
    voc_class_name,
    voc_class_names,
    BoostConfig,
    ByteTrackConfig,
    ByteTracker,
    TrackingPipeline,
    JpegDecoder,
    JpegEncoder,
    Track,
    VideoDecoder,
    VideoEncoder,
    VideoFrame,
    cosine_distance,
    cosine_similarity,
    dequantize,
    float_to_half,
    euclidean_distance,
    mpp_available,
    nms,
    normalize_embedding,
    reid_preprocess,
    rga_available,
    rga_version,
)

__version__ = rcdl_py.__version__

__all__ = [
    "Engine",
    "Detection",
    "DetectionPipeline",
    "AsyncVideoDetectionPipeline",
    "DmaBuf",
    "DmaHeap",
    "NpuCore",
    "coco_class_name",
    "coco_class_names",
    "compute_letterbox",
    "decode",
    "decode_yolo_ltrb",
    "dequantize",
    "float_to_half",
    "nms",
    "rga_available",
    "rga_version",
    "mpp_available",
    "VideoDecoder",
    "VideoEncoder",
    "VideoFrame",
    "JpegEncoder",
    "JpegDecoder",
    "decode_video",
    "ByteTracker",
    "TrackingPipeline",
    "track",
    "ByteTrackConfig",
    "BoostConfig",
    "Track",
    "cosine_similarity",
    "normalize_embedding",
    "reid_preprocess",
    "letterbox",
    "cvt_color",
    "detect",
    # tasks: classification
    "Classifier",
    "ClsConfig",
    "ClsResult",
    "CropBox",
    "decode_classification",
    "class_count_from_shape",
    "class_label",
    "center_crop_box",
    "load_class_labels",
    "looks_like_probabilities",
    "classify",
    # tasks: embeddings
    "EmbedConfig",
    "EmbedMatch",
    "EmbeddingBank",
    "ImageEmbedder",
    "decode_embedding",
    "embedding_dim_from_shape",
    "cosine_distance",
    "euclidean_distance",
    "embed",
    # tasks: instance segmentation
    "InstanceMask",
    "InstanceSegConfig",
    "InstanceSegmenter",
    "decode_instance_seg",
    "segment_instances",
    # tasks: semantic segmentation
    "SegMask",
    "Segmenter",
    "decode_seg",
    "seg_colorize",
    "seg_resize",
    "seg_to_source",
    "voc_class_name",
    "voc_class_names",
    "segment",
    # tasks: monocular depth
    "DepthConfig",
    "DepthEstimator",
    "DepthMap",
    "decode_depth",
    "depth_colorize",
    "depth_resize",
    "depth_to_gray8",
    "depth_to_source",
    "estimate_depth",
    # tasks: pose
    "Keypoint",
    "PoseConfig",
    "PoseDetection",
    "PoseEstimator",
    "decode_pose",
    "coco_keypoint_name",
    "coco_keypoint_names",
    "coco_skeleton",
    "estimate_pose",
    # tasks: oriented bounding boxes
    "ObbConfig",
    "ObbDetection",
    "ObbDetector",
    "RotatedBox",
    "decode_obb",
    "rotated_iou",
    "rotated_nms",
    "dota_class_name",
    "dota_class_names",
    "detect_obb",
    # tasks: OCR
    "OcrDetConfig",
    "OcrRecConfig",
    "TextBox",
    "TextDetector",
    "TextLine",
    "TextRecognizer",
    "TextAngleClassifier",
    "TextOrientation",
    "decode_text_boxes",
    "decode_text_orientation",
    "ocr_line_fit_width",
    "sort_text_boxes",
    "min_area_quad",
    "unclip_quad",
    "ctc_greedy_decode",
    "load_char_dict",
    "detect_text",
    "recognize_text",
    # tasks: face detection
    "FaceConfig",
    "FaceDetection",
    "FaceDetector",
    "FaceHeadLayout",
    "FaceRecognizer",
    "embed_face",
    "decode_faces",
    "arcface_template",
    "face_align_transform",
    "similarity_transform",
    "generate_priors",
    "detect_faces",
    # tasks: sparse local features
    "FeatureExtractor",
    "FeatureSet",
    "XfeatConfig",
    "xfeat_preprocess",
    "decode_xfeat",
    "match_features",
    "extract_features",
    # tasks: whole-body pose
    "BodyPart",
    "CropRect",
    "WholeBodyEstimator",
    "body_part",
    "body_part_name",
    "body_part_range",
    "crop_geometry",
    "decode_simcc",
    "estimate_wholebody",
    # tasks: promptable segmentation
    "PromptMask",
    "PromptableSegmenter",
    "encode_box_prompt",
    "encode_point_prompt",
    "mask_from_logits",
    "prompt_mask",
    # tasks: optical flow
    "OpticalFlowEstimator",
    "decode_flow",
    "flow_colorize",
    "flow_endpoint_error",
    "flow_preprocess",
    "estimate_flow",
    # tasks: super-resolution
    "SuperResConfig",
    "SuperResolver",
    "plan_tiles",
    "tile_weight",
    "upscale",
    # tasks: open-vocabulary detection
    "LabelMap",
    "yolo_head_classes",
    # tasks: panoptic driving (anchor-based head)
    "Anchor",
    "AnchorDetectConfig",
    "AnchorDetector",
    "decode_yolov5_anchor",
    "__version__",
    # speech
    "FbankConfig",
    "Fbank",
    "lfr_stack",
    "Cmvn",
    "parse_kaldi_cmvn",
    "load_kaldi_cmvn",
    "CifConfig",
    "cif_integrate",
    "load_token_list",
    "argmax_tokens",
    "tokens_to_text",
    "AsrResult",
    "ParaformerConfig",
    "ParaformerAsr",
    "to_pinyin",
    "CommandMatch",
    "CommandMatcher",
    "SileroVad",
    "VadConfig",
    "VadSegment",
    "VadSegmenter",
    "KwsConfig",
    "KwsDetection",
    "KeywordSpotter",
    "IstftConfig",
    "Istft",
    "VocosConfig",
    "VocosVocoder",
    "MatchaFrontendConfig",
    "MatchaFrontend",
    "MatchaConfig",
    "MatchaResult",
    "MatchaAcoustic",
    "TtsResult",
    "TextToSpeech",
    "TextToSpeechConfig",
    "scale_silence",
]


class Engine:
    """Load an ``.rknn`` model and run NPU inference with numpy in/out.

    Inputs are given as they come off an image pipeline — ``uint8`` HWC arrays
    for quantized models (the toolkit folds mean/std into the graph), ``float32``
    for float models. Outputs come back as float32 arrays in the model's own
    shape, already dequantized.
    """

    def __init__(self, path: str, core: NpuCore = NpuCore.AUTO, init_flags: int = 0,
                 float_inputs: Sequence[int] = (), custom_ops: bool = True):
        """``float_inputs`` lists inputs whose tensor is a normalized MAP rather
        than image bytes (XFeat's InstanceNorm output). Those are handed to the
        runtime as float32; the u8 path a quantized model normally uses has no
        negative range to put half of such a map in, and clips it silently.

        ``custom_ops`` registers RCDL's CPU kernels for operators librknnrt does
        not implement — currently ``GridSample``, which every correlation-based
        optical-flow network needs. It is on by default because registering an
        unused type costs nothing; turn it off only to show that a model that
        fails at ``run()`` fails for that reason."""
        self._e = rcdl_py.Engine(path, core, init_flags, list(float_inputs), custom_ops)

    @classmethod
    def _wrap(cls, raw: "rcdl_py.Engine") -> "Engine":
        obj = cls.__new__(cls)
        obj._e = raw
        return obj

    def dup(self, core: NpuCore = NpuCore.AUTO) -> "Engine":
        """A second context sharing this model's weights, optionally pinned to
        another NPU core (the way to use all three RK3588 cores concurrently)."""
        return Engine._wrap(self._e.dup(core))

    # --- introspection ---------------------------------------------------
    @property
    def path(self) -> str:
        return self._e.path

    @property
    def core(self) -> NpuCore:
        return self._e.core

    @property
    def num_inputs(self) -> int:
        return self._e.num_inputs

    @property
    def num_outputs(self) -> int:
        return self._e.num_outputs

    def input_shape(self, i: int) -> list[int]:
        return self._e.input_shape(i)

    def output_shape(self, i: int) -> list[int]:
        return self._e.output_shape(i)

    def input_name(self, i: int) -> str:
        return self._e.input_name(i)

    def output_name(self, i: int) -> str:
        return self._e.output_name(i)

    def input_dtype(self, i: int) -> np.dtype:
        return _np_dtype(self._e.input_dtype(i))

    def output_dtype(self, i: int) -> np.dtype:
        return _np_dtype(self._e.output_dtype(i))

    def input_format(self, i: int) -> str:
        """'NHWC' / 'NCHW' — the layout input i is provided in."""
        return self._e.input_format(i)

    def input_bytes(self, i: int) -> int:
        return self._e.input_bytes(i)

    def input_packed_bytes(self, i: int) -> int:
        return self._e.input_packed_bytes(i)

    def input_width_stride(self, i: int) -> int:
        return self._e.input_width_stride(i)

    def output_bytes(self, i: int) -> int:
        return self._e.output_bytes(i)

    def output_quant(self, i: int) -> tuple[int, int, float, int]:
        """(qnt_type, zero_point, scale, fl) of output i."""
        return self._e.output_quant(i)

    # --- data path -------------------------------------------------------
    def set_input(self, i: int, array: np.ndarray) -> None:
        arr = np.ascontiguousarray(array)
        want = self.input_dtype(i)
        if arr.dtype != want:
            arr = arr.astype(want, copy=False)
        self._e.set_input(i, arr)

    def run(self) -> None:
        """Run one inference on the inputs set so far (blocking, GIL released)."""
        self._e.infer()

    def output(self, i: int) -> np.ndarray:
        """Output i as a dequantized float32 array in the model's shape."""
        return self._e.output_float(i)

    def output_raw(self, i: int) -> bytes:
        return self._e.output_raw(i)

    def infer(self, inputs: Sequence[np.ndarray] | np.ndarray) -> list[np.ndarray]:
        """Set every input, run, and return all outputs as float32 arrays."""
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        if len(inputs) != self.num_inputs:
            raise ValueError(f"expected {self.num_inputs} inputs, got {len(inputs)}")
        for i, a in enumerate(inputs):
            self.set_input(i, a)
        self.run()
        return [self.output(i) for i in range(self.num_outputs)]

    # --- task pipelines --------------------------------------------------
    def detector(self, **kwargs) -> DetectionPipeline:
        """A DetectionPipeline driving this Engine (RGA letterbox -> NPU -> NMS).

        Keyword arguments mirror ``PipelineConfig``: ``model_input``
        ("rgb888"/"bgr888" — the channel order the model was built with),
        ``conf_thresh``, ``iou_thresh``, ``max_dets``, ``num_classes``,
        ``apply_sigmoid``, ``pad``, ``backend``, and for YUV sources
        ``studio_range`` / ``matrix`` ("bt601" or "bt709", see :func:`letterbox`).
        """
        return DetectionPipeline(self._e, **kwargs)

    def tracker(self, reid: "Engine | None" = None, **kwargs) -> TrackingPipeline:
        """A TrackingPipeline driving this Engine (detect + ByteTrack).

        Pass ``reid`` — a second :class:`Engine` holding an appearance model —
        to associate on geometry AND appearance, which is what holds identities
        through the occlusions motion alone loses. Its input shape supplies the
        crop size; ``reid_min_score`` and ``reid_max_crops`` bound how many
        crops a frame may embed, because that is what makes frame time scale
        with crowd size.

        The remaining keyword arguments are :meth:`detector`'s, plus
        ``track_config`` (a :class:`ByteTrackConfig`).
        """
        return TrackingPipeline(self._e, None if reid is None else reid._e, **kwargs)

    def video_detector(self, codec: str = "h264", **kwargs) -> AsyncVideoDetectionPipeline:
        """Compressed video in, detections out — the whole VPU/RGA/NPU path in C++.

        Feed raw elementary-stream bytes with ``submit(data)`` and take results
        with ``next()`` / ``try_next()``; the decode, letterbox and inference
        stages run on C++ threads with the GIL released, so a Python driver that
        only pumps bytes still reaches the C++ throughput. Chunks may be any
        size — MPP's parser finds the access units.

        ``submit()`` returning False is BACK-PRESSURE, not an error: the same
        bytes must be offered again after draining results (see ``.finished``
        for the other reason it can be False). Blocking there instead would
        deadlock a single-threaded driver against its own pipeline.

            p = engine.video_detector(codec="h264")
            for chunk in chunks:
                while not p.submit(chunk) and not p.finished:
                    while (d := p.try_next()) is not None:
                        use(d)
            p.finish()
            while (d := p.next()) is not None:
                use(d)

        Keyword arguments are :meth:`detector`'s, plus ``workers`` /
        ``pin_cores`` / ``reorder_depth`` (NPU contexts), ``queue_depth``
        (decoded frames buffered between the VPU and RGA) and the decoder's
        ``external_buffers`` / ``extra_buffers``.
        """
        return AsyncVideoDetectionPipeline(self._e, codec=codec, **kwargs)

    def classifier(self, **kwargs) -> Classifier:
        """A Classifier driving this Engine (RGA centre-crop -> NPU -> top-k).

        Keyword arguments: ``top_k``, ``apply_softmax``, ``model_input``,
        ``crop_ratio`` (0.875 is the ImageNet eval convention, 1.0 a plain
        resize), ``backend``, ``output_index``.
        """
        return Classifier(self._e, **kwargs)

    def embedder(self, **kwargs) -> ImageEmbedder:
        """An ImageEmbedder driving this Engine (RGA crop -> NPU -> one vector).

        Keyword arguments: ``l2_normalize``, ``model_input``, ``box_expand``,
        ``backend``, ``output_index``.
        """
        return ImageEmbedder(self._e, **kwargs)

    def feature_extractor(self, **kwargs) -> FeatureExtractor:
        """A FeatureExtractor driving this Engine (XFeat sparse features).

        Keyword arguments: ``config`` (an :class:`XfeatConfig`), ``output_base``.
        The Engine must have been opened with ``float_inputs=[0]`` — this model's
        input is a normalized map, not pixels — and the extractor says so rather
        than running if it was not.
        """
        return FeatureExtractor(self._e, **kwargs)

    def wholebody_estimator(self, **kwargs) -> WholeBodyEstimator:
        """A WholeBodyEstimator driving this Engine (133 keypoints, top-down).

        Keyword arguments: ``kpt_thresh``, ``padding`` (box padding before the
        aspect fix), ``split_ratio``, ``pad``. One inference per person, so a
        detector runs first — see :func:`estimate_wholebody`.
        """
        return WholeBodyEstimator(self._e, **kwargs)

    def prompt_segmenter(self, decoder: "Engine", **kwargs) -> PromptableSegmenter:
        """A PromptableSegmenter over this Engine (the SAM encoder) and a decoder.

        Keyword arguments: ``mask_thresh``, ``multimask``, ``pad``. The encoder
        runs once per frame in ``set_image``; every ``box`` / ``point`` after it
        is a decoder pass against the same embedding.
        """
        return PromptableSegmenter(self._e, decoder._e, **kwargs)

    def flow_estimator(self, **kwargs) -> OpticalFlowEstimator:
        """An OpticalFlowEstimator driving this Engine (two frames -> a field).

        Keyword arguments: ``output_index``, ``input0_index``, ``input1_index``.
        The model's input size comes from the Engine, and the returned field is
        already in the source image's pixels.
        """
        return OpticalFlowEstimator(self._e, **kwargs)

    def upscaler(self, **kwargs) -> SuperResolver:
        """A SuperResolver driving this Engine (tiled x4 upscaling).

        Keyword arguments: ``overlap`` (input pixels of cross-fade between
        neighbouring tiles), ``input_index``, ``output_index``. The scale factor
        and tile size come from the model's own shapes.
        """
        return SuperResolver(self._e, **kwargs)

    def instance_segmenter(self, **kwargs) -> InstanceSegmenter:
        """An InstanceSegmenter driving this Engine (YOLO-seg head).

        Grids, class and coefficient counts, reg_max and the channel orders come
        from the model; the keyword arguments are the thresholds (``conf_thresh``,
        ``iou_thresh``, ``max_dets``, ``apply_sigmoid``) and the mask options
        (``mask_thresh``, ``compute_masks``, ``full_frame_masks``).
        """
        return InstanceSegmenter(self._e, **kwargs)

    def segmenter(self, **kwargs) -> Segmenter:
        """A Segmenter driving this Engine (semantic segmentation).

        Keyword arguments: ``num_classes``, ``argmaxed``, ``score``
        ("none"/"softmax"/"max"), ``model_input``, ``output_index``. The channel
        order comes from the output tensor itself, not from a flag.
        """
        return Segmenter(self._e, **kwargs)

    def anchor_detector(self, config: "AnchorDetectConfig | None" = None,
                        output_base: int = 0) -> AnchorDetector:
        """An AnchorDetector driving this Engine (anchor-based YOLOv5-style head).

        Unlike the other factories this one does NO preprocessing: a panoptic
        driving model feeds three heads off one inference, so something else
        (a ``Segmenter`` on one of the mask outputs) runs the frame and this
        decodes the raw detection outputs that inference left behind.

        ``config`` defaults to the YOLOP prior set; the priors are part of the
        model, not a tuning knob (see docs/MODELS.md).
        """
        return AnchorDetector(self._e, config if config is not None else AnchorDetectConfig(),
                              output_base)

    def label_map(self, path: str | None = None) -> LabelMap:
        """The open-vocabulary label table for this model.

        Defaults to ``<model>.labels.txt`` beside the ``.rknn``, and checks it
        against the class count the model declares — a table from a different
        vocabulary renames every detection without changing a single box.
        """
        if path is None:
            stem = self.path[: -len(".rknn")] if self.path.endswith(".rknn") else self.path
            path = stem + ".labels.txt"
        lm = LabelMap.from_file(path)
        # The vocabulary's length is the CLAIM: a model with no branch that wide
        # fails to resolve at all, which is the mismatch worth catching, and
        # yolo_head_classes rejects a claim that only "fits" by reinterpreting
        # the class branch as a box head.
        lm.require_size(yolo_head_classes(self._e, len(lm)))
        return lm

    def face_recognizer(self, **kwargs) -> FaceRecognizer:
        """A FaceRecognizer driving this Engine (identity embedding).

        Keyword arguments: ``model_input`` (channel order the model was trained
        on — ArcFace is rgb888), ``normalize``, ``pad``, ``output_index``.
        """
        return FaceRecognizer(self._e, **kwargs)

    def depth_estimator(self, **kwargs) -> DepthEstimator:
        """A DepthEstimator driving this Engine (monocular depth).

        Keyword arguments mirror ``DepthConfig``: ``scale``, ``shift``,
        ``inverse``, ``clip_lo``, ``clip_hi``, ``normalize``, plus
        ``model_input`` and ``output_index``.
        """
        return DepthEstimator(self._e, **kwargs)

    def pose_estimator(self, **kwargs) -> PoseEstimator:
        """A PoseEstimator driving this Engine (YOLO pose head).

        Grids, keypoint count, reg_max, channel order and strides come from the
        model; the keyword arguments are the thresholds (``conf_thresh``,
        ``iou_thresh``, ``max_dets``) and the activation conventions
        (``apply_sigmoid``, ``kpt_decode``, ``kpt_apply_sigmoid``).
        """
        return PoseEstimator(self._e, **kwargs)

    def obb_detector(self, **kwargs) -> ObbDetector:
        """An ObbDetector driving this Engine (YOLO oriented-box head).

        Keyword arguments: ``conf_thresh``, ``iou_thresh`` (rotated IoU, so
        lower than the axis-aligned default), ``max_dets``, ``apply_sigmoid``,
        ``apply_angle_sigmoid``, ``angle_bias``, ``regularize``.
        """
        return ObbDetector(self._e, **kwargs)

    def text_detector(self, **kwargs) -> TextDetector:
        """A TextDetector driving this Engine (PP-OCR DBNet head).

        Keyword arguments: ``config`` (an :class:`OcrDetConfig`),
        ``model_input``, ``pad``, ``backend``, ``output_index``.
        """
        return TextDetector(self._e, **kwargs)

    def text_angle_classifier(self, **kwargs) -> TextAngleClassifier:
        """A TextAngleClassifier driving this Engine (PP-OCR 0/180 direction head).

        It belongs between :meth:`text_detector` and :meth:`text_recognizer`,
        and it is not optional decoration: a CTC recogniser fed an upside-down
        line does not fail, it returns confident nonsense. Feed it the same crop
        you are about to recognise, and rotate the crop 180 degrees when the
        result's ``flip180`` is set.

        Keyword arguments: ``thresh`` (the flip gate, PP-OCR's default 0.9),
        ``model_input``, ``pad``, ``backend``, ``output_index``.
        """
        return TextAngleClassifier(self._e, **kwargs)

    def text_recognizer(self, dict_path, **kwargs) -> TextRecognizer:
        """A TextRecognizer driving this Engine (CRNN + CTC).

        ``dict_path`` is a character dictionary file, or an already-loaded list
        of tokens (the right form when several Engines share one 6625-entry
        table). Keyword arguments: ``config`` (an :class:`OcrRecConfig`),
        ``paddle_special`` (path form only), ``model_input``, ``fit``
        (``"pad"``, the default and PP-OCR's own rule — scale to the model's
        height, cap the width there, anchor left and pad the remainder;
        ``"stretch"``, which is the same thing for any crop wider than the
        input's aspect ratio and squashes anything narrower; or
        ``"letterbox"``),
        ``input_scale`` / ``input_shift``, ``backend``, ``output_index``.

        Unlike the other heads this one preprocesses on the host, because the
        deployed recognition export is a FLOAT model and so has no uint8 input
        tensor for RGA to write into. ``input_scale`` / ``input_shift`` map a
        pixel to that float input (the default 1/255 puts it in [0, 1]); a
        quantized rec export takes the bytes unchanged and ignores both.
        """
        return TextRecognizer(self._e, dict_path, **kwargs)

    def face_detector(self, **kwargs) -> FaceDetector:
        """A FaceDetector driving this Engine (RetinaFace).

        The canvas and anchor count come from the model, and the generated
        priors are cross-checked against them at construction. Keyword
        arguments: ``conf_thresh``, ``iou_thresh``, ``max_faces``,
        ``apply_softmax``, ``config`` (a :class:`FaceConfig`, for a non-default
        prior recipe), ``model_input``, ``backend``.

        ``model_input`` defaults to "bgr888", not "rgb888" like the YOLO heads:
        RetinaFace is trained with BGR channel order and BGR mean subtraction.
        """
        return FaceDetector(self._e, **kwargs)

    # --- diagnostics ----------------------------------------------------
    def last_run_micros(self) -> int:
        return self._e.last_run_micros()

    def perf_detail(self) -> str:
        return self._e.perf_detail()

    def sdk_version(self) -> str:
        return self._e.sdk_version()

    def driver_version(self) -> str:
        return self._e.driver_version()


_DTYPES = {
    "f32": np.float32,
    "f16": np.float16,
    "i8": np.int8,
    "u8": np.uint8,
    "i16": np.int16,
    "u16": np.uint16,
    "i32": np.int32,
    "u32": np.uint32,
    "i64": np.int64,
    "bool": np.bool_,
}


def _np_dtype(name: str) -> np.dtype:
    try:
        return np.dtype(_DTYPES[name])
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype {name!r}") from exc


# --------------------------------------------------------------------------- #
# Preprocessing conveniences                                                   #
# --------------------------------------------------------------------------- #
# The compiled helpers take a flat byte buffer plus an explicit geometry because
# a numpy array cannot express a padded row stride or a semi-planar YUV layout.
# These wrappers do the shaping so callers stay in HxWxC numpy land.

_PACKED_CHANNELS = {"rgb888": 3, "bgr888": 3, "rgba8888": 4, "bgra8888": 4, "gray8": 1}


def _as_buffer(img: np.ndarray, fmt: str) -> tuple[np.ndarray, int, int]:
    """Flatten an image array to (bytes, width, height) for the compiled layer."""
    a = np.ascontiguousarray(img, dtype=np.uint8)
    if fmt in _PACKED_CHANNELS:
        if a.ndim == 2:
            h, w = a.shape
        elif a.ndim == 3:
            h, w = a.shape[0], a.shape[1]
            if a.shape[2] != _PACKED_CHANNELS[fmt]:
                raise ValueError(f"{fmt} expects {_PACKED_CHANNELS[fmt]} channels, got {a.shape[2]}")
        else:
            raise ValueError(f"{fmt} expects a 2-D or 3-D array, got {a.ndim}-D")
        return a.reshape(-1), w, h
    # semi-planar / planar YUV: (H*3//2, W) or flat with the size given separately
    if a.ndim != 2:
        raise ValueError(f"{fmt} expects a 2-D (H*3//2, W) array")
    rows, w = a.shape
    return a.reshape(-1), w, rows * 2 // 3


def _unpack(buf: np.ndarray, w: int, h: int, wstride: int, fmt: str) -> np.ndarray:
    """Shape a flat destination buffer back into a numpy image, dropping padding."""
    if fmt in _PACKED_CHANNELS:
        c = _PACKED_CHANNELS[fmt]
        img = buf.reshape(h, wstride, c)[:, :w, :]
        return img[:, :, 0] if c == 1 else img
    return buf.reshape(h * 3 // 2, wstride)[:, :w]


def letterbox(img, dst_w, dst_h, src_fmt="bgr888", dst_fmt="rgb888", pad=114,
              backend="auto", studio_range=True, matrix="bt601"):
    """Aspect-preserving letterbox. Returns (image, letterbox_geometry, backend).

    ``backend`` is "auto" (RGA when the hardware accepts the request, CPU
    otherwise), "rga" (raise if it cannot) or "cpu".

    ``studio_range`` and ``matrix`` ("bt601" or "bt709") describe the YUV side
    of any colour conversion the formats imply. RGA does YUV -> RGB in BT.601
    and BT.709 limited range, and RGB -> YUV in BT.601; "auto" sends the other
    combinations to the CPU.
    """
    flat, w, h = _as_buffer(img, src_fmt)
    buf, lb, used, wstride = rcdl_py.letterbox(flat, w, h, src_fmt, dst_w, dst_h, dst_fmt,
                                               pad, backend, studio_range, matrix)
    return _unpack(buf, dst_w, dst_h, wstride, dst_fmt), lb, used


def cvt_color(img, src_fmt, dst_fmt, backend="auto", studio_range=True, matrix="bt601"):
    """Colour-space conversion at the same size. Returns (image, backend).

    ``studio_range`` / ``matrix`` as in :func:`letterbox`.
    """
    flat, w, h = _as_buffer(img, src_fmt)
    buf, used, wstride = rcdl_py.cvt_color(flat, w, h, src_fmt, dst_fmt, backend, studio_range,
                                           matrix)
    return _unpack(buf, w, h, wstride, dst_fmt), used


def detect(pipeline: DetectionPipeline, img, fmt: str = "bgr888"):
    """Run a DetectionPipeline on a numpy image (BGR HxWx3 by default)."""
    flat, w, h = _as_buffer(img, fmt)
    return pipeline.process(flat, w, h, fmt)


# The task heads take the same (buffer, width, height, format) shape as the
# detection pipeline, because a numpy array cannot express a padded row stride
# or a semi-planar YUV layout. These wrappers do the flattening so callers stay
# in HxWxC numpy land, exactly as `detect()` does.


def track(pipeline: TrackingPipeline, img, fmt: str = "bgr888"):
    """Detect and associate one numpy frame. Returns this frame's Tracks, each
    with a stable ``track_id``, in source pixels."""
    flat, w, h = _as_buffer(img, fmt)
    return pipeline.process(flat, w, h, fmt)


def classify(classifier: Classifier, img, fmt: str = "bgr888"):
    """Top-k classify a numpy image. Returns ClsResults, best first."""
    flat, w, h = _as_buffer(img, fmt)
    return classifier.classify(flat, w, h, fmt)


def embed(embedder: ImageEmbedder, img, box=None, fmt: str = "bgr888") -> np.ndarray:
    """Embed a numpy image, or the ``box`` = (x1, y1, x2, y2) crop of it."""
    flat, w, h = _as_buffer(img, fmt)
    x1, y1, x2, y2 = box if box is not None else (0.0, 0.0, 0.0, 0.0)
    return embedder.embed(flat, w, h, fmt, x1, y1, x2, y2)


def extract_features(extractor: FeatureExtractor, img) -> FeatureSet:
    """Sparse features from a numpy BGR image (HxWx3 uint8).

    Unlike every other task helper here there is no ``fmt``: XFeat takes the
    plain channel MEAN as its grey, so the channel order does not change the
    result — but the array still has to be BGR-shaped 3-channel uint8, and a
    non-contiguous crop has to be copied first.
    """
    a = np.ascontiguousarray(img)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError("extract_features: expected an HxWx3 uint8 image")
    return extractor.extract(a)


def estimate_wholebody(estimator: WholeBodyEstimator, img, box, fmt: str = "bgr888") -> np.ndarray:
    """133 keypoints for one person's ``box`` = (x1, y1, x2, y2) -> (K, 3) array.

    One inference per person: loop the boxes, not the frames.
    """
    flat, w, h = _as_buffer(img, fmt)
    return estimator.estimate(flat, w, h, fmt, *box)


def prompt_mask(segmenter: PromptableSegmenter, img, box=None, point=None,
                positive: bool = True, fmt: str = "bgr888") -> PromptMask:
    """Encode a numpy image and prompt it once — the convenience path.

    Prompting the SAME image repeatedly should call ``set_image`` once and then
    ``box``/``point`` directly: the encoder is most of the cost, and this helper
    pays it every time by design.
    """
    flat, w, h = _as_buffer(img, fmt)
    segmenter.set_image(flat, w, h, fmt)
    if box is not None:
        return segmenter.box(*box)
    if point is None:
        raise ValueError("prompt_mask: give either box=(x1,y1,x2,y2) or point=(x,y)")
    return segmenter.point(point[0], point[1], positive)


def estimate_flow(estimator: OpticalFlowEstimator, a, b) -> np.ndarray:
    """Dense flow between two numpy BGR frames -> (H, W, 2) in source pixels."""
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    for x in (a, b):
        if x.ndim != 3 or x.shape[2] != 3 or x.dtype != np.uint8:
            raise ValueError("estimate_flow: expected HxWx3 uint8 BGR frames")
    return estimator.estimate(a, b)


def upscale(upscaler: SuperResolver, img) -> np.ndarray:
    """Upscale a numpy BGR image (HxWx3 uint8) by the model's own factor.

    Tiling is internal; ``upscaler.last_tile_count`` says how many inferences it
    took, and the cost is linear in that.
    """
    a = np.ascontiguousarray(img)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError("upscale: expected an HxWx3 uint8 BGR image")
    return upscaler.upscale(a)


def segment_instances(segmenter: InstanceSegmenter, img, fmt: str = "bgr888"):
    """Instance-segment a numpy image. Boxes and masks come back in source pixels."""
    flat, w, h = _as_buffer(img, fmt)
    return segmenter.process(flat, w, h, fmt)


def segment(segmenter: Segmenter, img, fmt: str = "bgr888") -> SegMask:
    """Semantically segment a numpy image; the label map is in source pixels."""
    flat, w, h = _as_buffer(img, fmt)
    return segmenter.process(flat, w, h, fmt)


def estimate_depth(estimator: DepthEstimator, img, fmt: str = "bgr888") -> DepthMap:
    """Estimate depth for a numpy image; the map is in source pixels."""
    flat, w, h = _as_buffer(img, fmt)
    return estimator.process(flat, w, h, fmt)


def estimate_pose(estimator: PoseEstimator, img, fmt: str = "bgr888"):
    """Estimate poses in a numpy image. Boxes and joints in source pixels."""
    flat, w, h = _as_buffer(img, fmt)
    return estimator.process(flat, w, h, fmt)


def detect_obb(detector: ObbDetector, img, fmt: str = "bgr888"):
    """Detect oriented boxes in a numpy image. Returns ObbDetections."""
    flat, w, h = _as_buffer(img, fmt)
    return detector.process(flat, w, h, fmt)


def detect_text(detector: TextDetector, img, fmt: str = "bgr888"):
    """Detect text quadrilaterals in a numpy image. Returns TextBoxes."""
    flat, w, h = _as_buffer(img, fmt)
    return detector.process(flat, w, h, fmt)


def recognize_text(recognizer: TextRecognizer, img, fmt: str = "bgr888") -> TextLine:
    """Recognize ONE already-cropped, upright text line.

    The step between :func:`detect_text` and here — crop each quadrilateral out
    of the original frame and warp it upright — is the caller's, because it is a
    host image operation and not a decode. A ``TextBox.pts`` is TL, TR, BR, BL,
    which is the order a 4-point perspective transform expects.
    """
    flat, w, h = _as_buffer(img, fmt)
    return recognizer.process(flat, w, h, fmt)


def embed_face(recognizer: FaceRecognizer, img, landmarks, fmt: str = "bgr888") -> np.ndarray:
    """Identity embedding for one face, given its five landmarks in SOURCE pixels.

    ``landmarks`` is a (5, 2) or (10,) float32 array in the order the detector
    produces: left eye, right eye, nose, left mouth corner, right mouth corner.
    The alignment warp happens inside — see docs/MODELS.md for why that is not
    left to the caller.
    """
    flat, w, h = _as_buffer(img, fmt)
    return recognizer.embed(flat, w, h, fmt, np.ascontiguousarray(landmarks, np.float32))


def detect_faces(detector: FaceDetector, img, fmt: str = "bgr888"):
    """Detect faces in a numpy image. Boxes and landmarks in source pixels."""
    flat, w, h = _as_buffer(img, fmt)
    return detector.process(flat, w, h, fmt)


def decode_video(path: str, codec: str | None = None, max_frames: int | None = None,
                 chunk_bytes: int = 1 << 16, **kwargs):
    """Iterate decoded frames of an elementary stream file.

    Yields :class:`VideoFrame` objects that still live in the VPU's dma-bufs, so
    each one is only valid until the next iteration — copy what you need with
    ``frame.to_numpy()``, or hand ``frame`` straight to
    ``DetectionPipeline.process_frame`` for the zero-copy path.

    ``codec`` defaults to a guess from the file extension.
    """
    import os

    if codec is None:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        codec = {"264": "h264", "h264": "h264", "avc": "h264",
                 "265": "h265", "h265": "h265", "hevc": "h265",
                 "vp9": "vp9", "av1": "av1", "ivf": "av1"}.get(ext, "h264")

    dec = VideoDecoder(codec=codec, **kwargs)
    n = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            while not dec.feed(chunk):
                f = dec.receive(5)
                if f is not None:
                    yield f
                    n += 1
                    if max_frames is not None and n >= max_frames:
                        return
            while True:
                f = dec.receive(0)
                if f is None:
                    break
                yield f
                n += 1
                if max_frames is not None and n >= max_frames:
                    return
    while True:
        f = dec.flush()
        if f is None:
            break
        yield f
        n += 1
        if max_frames is not None and n >= max_frames:
            return


# --- speech: Kaldi fbank front end + Paraformer ASR --------------------------
# The front end is the C++ one, bound, rather than a numpy re-implementation:
# a model trained on kaldi.fbank features transcribes worse — silently — on
# features that are merely similar, so there is exactly one definition of them.
FbankConfig = rcdl_py.FbankConfig
Fbank = rcdl_py.Fbank
lfr_stack = rcdl_py.lfr_stack
Cmvn = rcdl_py.Cmvn
parse_kaldi_cmvn = rcdl_py.parse_kaldi_cmvn
load_kaldi_cmvn = rcdl_py.load_kaldi_cmvn
CifConfig = rcdl_py.CifConfig
cif_integrate = rcdl_py.cif_integrate
load_token_list = rcdl_py.load_token_list
argmax_tokens = rcdl_py.argmax_tokens
tokens_to_text = rcdl_py.tokens_to_text
AsrResult = rcdl_py.AsrResult
ParaformerConfig = rcdl_py.ParaformerConfig
to_pinyin = rcdl_py.to_pinyin
CommandMatch = rcdl_py.CommandMatch
CommandMatcher = rcdl_py.CommandMatcher
VadConfig = rcdl_py.VadConfig
KwsConfig = rcdl_py.KwsConfig
KwsDetection = rcdl_py.KwsDetection
VadSegment = rcdl_py.VadSegment
IstftConfig = rcdl_py.IstftConfig
Istft = rcdl_py.Istft
VocosConfig = rcdl_py.VocosConfig
MatchaFrontendConfig = rcdl_py.MatchaFrontendConfig
MatchaFrontend = rcdl_py.MatchaFrontend
MatchaConfig = rcdl_py.MatchaConfig
MatchaResult = rcdl_py.MatchaResult
TtsResult = rcdl_py.TtsResult
TextToSpeechConfig = rcdl_py.TextToSpeechConfig


def scale_silence(pcm, sample_rate: int, scale: float, min_pause_s: float = 0.2,
                  level: float = 0.01) -> np.ndarray:
    """Shorten every pause (``|x| <= level`` for at least ``min_pause_s``) to
    ``scale`` of its length — sherpa-onnx's silence_scale. float32 PCM in and out."""
    return rcdl_py.scale_silence(np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1),
                                 sample_rate, scale, min_pause_s, level)


class ParaformerAsr:
    """Non-autoregressive speech recognition: one 16 kHz utterance in, text out.

    Three compiled stages (encoder, predictor, decoder) with continuous
    integrate-and-fire on the CPU between the last two. Feed ONE utterance of at
    most :attr:`max_seconds` — longer audio is truncated and flagged on the
    result, so segment it with a VAD first. There is no resampler: the PCM must
    already be 16 kHz mono float in [-1, 1]."""

    def __init__(self, encoder: "Engine", predictor: "Engine", decoder: "Engine",
                 vocab, cmvn: "Cmvn", config: "ParaformerConfig | None" = None):
        self.encoder, self.predictor, self.decoder = encoder, predictor, decoder
        self.config = config if config is not None else ParaformerConfig()
        self._a = rcdl_py.ParaformerAsr(encoder._e, predictor._e, decoder._e, list(vocab),
                                        cmvn, self.config)

    @classmethod
    def from_files(cls, encoder: str, predictor: str, decoder: str, tokens: str,
                   cmvn: str, config: "ParaformerConfig | None" = None) -> "ParaformerAsr":
        """Build from the three ``.rknn`` paths, a token list and an ``am.mvn``."""
        return cls(Engine(encoder), Engine(predictor), Engine(decoder),
                   load_token_list(tokens), load_kaldi_cmvn(cmvn), config)

    def transcribe(self, pcm: np.ndarray) -> "AsrResult":
        """Mono 16 kHz PCM (float in [-1, 1], or int16) -> AsrResult."""
        return self._a.transcribe(_as_float_pcm(pcm))

    def transcribe_features(self, feats: np.ndarray) -> "AsrResult":
        """(frames, feature_dim) LFR + CMVN features -> AsrResult."""
        return self._a.transcribe_features(np.ascontiguousarray(feats, dtype=np.float32))

    def features(self, pcm: np.ndarray) -> np.ndarray:
        """Front end only: PCM (float in [-1, 1], or int16) -> (frames, feature_dim)
        normalised features."""
        return self._a.features(_as_float_pcm(pcm))

    @property
    def max_seconds(self) -> float:
        return self._a.max_seconds

    @property
    def max_frames(self) -> int:
        return self._a.max_frames

    @property
    def feature_dim(self) -> int:
        return self._a.feature_dim

    @property
    def max_tokens(self) -> int:
        return self._a.max_tokens

    @property
    def vocab_size(self) -> int:
        return self._a.vocab_size


def _as_float_pcm(pcm) -> np.ndarray:
    """float in [-1, 1] as is; int16 rescaled to it (not just cast)."""
    pcm = np.asarray(pcm)
    if pcm.dtype == np.int16:
        pcm = pcm.astype(np.float32) / 32768.0
    return np.ascontiguousarray(pcm, dtype=np.float32).reshape(-1)


class SileroVad:
    """Silero VAD v6 on the NPU: speech probability per 512-sample (32 ms)
    chunk of 16 kHz audio. Streaming: the network's state and the previous
    chunk's tail carry over between calls, so feed one stream in order and
    :meth:`reset` between streams."""

    chunk = rcdl_py.SileroVad.chunk
    sample_rate = rcdl_py.SileroVad.sample_rate

    def __init__(self, engine: "Engine | str"):
        self.engine = Engine(engine) if isinstance(engine, str) else engine
        self._v = rcdl_py.SileroVad(self.engine._e)

    def probability(self, chunk) -> float:
        """512 samples (float in [-1, 1], or int16) -> speech probability."""
        return self._v.probability(_as_float_pcm(chunk))

    def probabilities(self, pcm) -> np.ndarray:
        """A whole recording from a reset -> one probability per full chunk."""
        pcm = _as_float_pcm(pcm)
        self.reset()
        n = len(pcm) // self.chunk
        return np.array([self._v.probability(pcm[i * self.chunk:(i + 1) * self.chunk])
                         for i in range(n)], dtype=np.float32)

    def reset(self) -> None:
        self._v.reset()


class VadSegmenter:
    """Cuts a 16 kHz stream into utterances with a :class:`SileroVad`.

    ``feed(pcm)`` takes audio of any length and returns the :class:`VadSegment`
    s that closed during it (``start``/``end`` in samples since the last reset,
    ``pcm`` ready for :meth:`ParaformerAsr.transcribe`); ``flush()`` ends the
    stream. ``segments(pcm)`` does a whole recording at once."""

    def __init__(self, vad: "SileroVad | Engine | str", config: "VadConfig | None" = None):
        self.vad = vad if isinstance(vad, SileroVad) else SileroVad(vad)
        self.config = config if config is not None else VadConfig()
        self._s = rcdl_py.VadSegmenter(self.vad._v, self.config)

    def feed(self, pcm) -> "list[VadSegment]":
        return self._s.feed(_as_float_pcm(pcm))

    def flush(self) -> "list[VadSegment]":
        return self._s.flush()

    def segments(self, pcm) -> "list[VadSegment]":
        return self._s.segments(_as_float_pcm(pcm))

    def reset(self) -> None:
        self._s.reset()

    @property
    def in_speech(self) -> bool:
        return self._s.in_speech

    @property
    def last_probability(self) -> float:
        return self._s.last_probability


class KeywordSpotter:
    """Wake words on the NPU: a streaming zipformer transducer (sherpa-onnx's
    KWS model) whose encoder runs one 160 ms chunk per call, with the decoder,
    joiner and keyword search on the CPU. Keywords are token lines in
    sherpa-onnx's format (``"n ǐ h ǎo x iǎo ch ē @你好小车"``), so a new wake
    word needs no new model. Streaming: feed one stream in order."""

    def __init__(self, encoder: "Engine | str", decoder_joiner: str, tokens: str,
                 keywords: "str | None" = None, config: "KwsConfig | None" = None):
        self.engine = Engine(encoder) if isinstance(encoder, str) else encoder
        self.config = config if config is not None else KwsConfig()
        self._k = rcdl_py.KeywordSpotter(self.engine._e, decoder_joiner, tokens, self.config)
        if keywords is not None:
            if "\n" not in keywords and os.path.isfile(keywords):
                self._k.load_keywords(keywords)
            else:
                self._k.set_keywords(keywords)

    def set_keywords(self, text: str) -> None:
        """Keyword lines (``tokens [:boost] [#threshold] @phrase``); resets the stream."""
        self._k.set_keywords(text)

    def load_keywords(self, path: str) -> None:
        self._k.load_keywords(path)

    @property
    def keywords(self) -> "list[str]":
        return self._k.keywords

    def feed(self, pcm) -> "list[KwsDetection]":
        """16 kHz PCM (float in [-1, 1], or int16), any length -> what fired."""
        return self._k.feed(_as_float_pcm(pcm))

    def flush(self) -> "list[KwsDetection]":
        """End of stream: pushes the last partial chunk through with silence,
        returns what fires, then resets. Call it when a recording ends."""
        return self._k.flush()

    def encode(self, pcm) -> np.ndarray:
        """Encoder output ``(frames, D)`` for a whole recording, from a reset."""
        return self._k.encode(_as_float_pcm(pcm))

    def reset(self) -> None:
        self._k.reset()

    def token_id(self, token: str) -> int:
        return self._k.token_id(token)

    @property
    def vocab_size(self) -> int:
        return self._k.vocab_size


class VocosVocoder:
    """Mel spectrogram -> waveform: Vocos on the NPU, inverse STFT on the CPU.

    The compiled model takes a fixed number of mel frames (:attr:`chunk_frames`);
    longer input is cut into overlapping chunks whose seams are exact. Feed the
    ``(num_mels, frames)`` log-mel an acoustic model such as Matcha-TTS emits;
    the result is float PCM at :attr:`sample_rate`, ``(frames - 1) * hop``
    samples long."""

    def __init__(self, engine: "Engine | str", config: "VocosConfig | None" = None):
        self.engine = Engine(engine) if isinstance(engine, str) else engine
        self.config = config if config is not None else VocosConfig()
        self._v = rcdl_py.VocosVocoder(self.engine._e, self.config)

    def synthesize(self, mel: np.ndarray) -> np.ndarray:
        """``(num_mels, frames)`` log-mel (a leading batch of 1 is accepted) -> float32 PCM."""
        mel = np.asarray(mel, dtype=np.float32)
        if mel.ndim == 3 and mel.shape[0] == 1:
            mel = mel[0]
        return self._v.synthesize(np.ascontiguousarray(mel))

    @property
    def num_mels(self) -> int:
        return self._v.num_mels

    @property
    def chunk_frames(self) -> int:
        return self._v.chunk_frames

    @property
    def body_frames(self) -> int:
        return self._v.body_frames

    @property
    def sample_rate(self) -> int:
        return self._v.sample_rate


class MatchaAcoustic:
    """Matcha-TTS acoustic model: one sentence's token ids -> log-mel.

    Two compiled graphs (encoder with the duration predictor; decoder with three
    flow-matching steps), length regulation on the CPU between them."""

    def __init__(self, encoder: "Engine | str", decoder: "Engine | str", embedding: str,
                 config: "MatchaConfig | None" = None):
        self.encoder = Engine(encoder) if isinstance(encoder, str) else encoder
        self.decoder = Engine(decoder) if isinstance(decoder, str) else decoder
        self.config = config if config is not None else MatchaConfig()
        self._a = rcdl_py.MatchaAcoustic(self.encoder._e, self.decoder._e, embedding,
                                         self.config)

    def synthesize(self, ids) -> "MatchaResult":
        return self._a.synthesize([int(i) for i in ids])

    def seed(self, seed: int) -> None:
        self._a.seed(seed)

    @property
    def max_tokens(self) -> int:
        return self._a.max_tokens

    @property
    def max_frames(self) -> int:
        return self._a.max_frames

    @property
    def num_mels(self) -> int:
        return self._a.num_mels

    @property
    def vocab_size(self) -> int:
        return self._a.vocab_size


class TextToSpeech:
    """Chinese text in, 16 kHz float PCM out — Matcha-TTS + Vocos, all on the board.

    ``TextToSpeech.from_files(encoder, decoder, embedding, vocoder, tokens, lexicon)``; then
    ``.synthesize(text).pcm``. Each sentence is synthesised on its own, the
    audio concatenated, and pauses longer than 0.2 s shortened to
    ``tts_config.silence_scale`` (0.2, as sherpa-onnx; 1 keeps them)."""

    def __init__(self, frontend: "MatchaFrontend", acoustic: "MatchaAcoustic",
                 vocoder: "VocosVocoder", tts_config: "TextToSpeechConfig | None" = None):
        self.frontend, self.acoustic, self.vocoder = frontend, acoustic, vocoder
        self.tts_config = tts_config if tts_config is not None else TextToSpeechConfig()
        self._t = rcdl_py.TextToSpeech(frontend, acoustic._a, vocoder._v, self.tts_config)

    @classmethod
    def from_files(cls, encoder: str, decoder: str, embedding: str, vocoder: str, tokens: str,
                   lexicon: str, config: "MatchaConfig | None" = None,
                   tts_config: "TextToSpeechConfig | None" = None) -> "TextToSpeech":
        return cls(MatchaFrontend(tokens, lexicon),
                   MatchaAcoustic(encoder, decoder, embedding, config), VocosVocoder(vocoder),
                   tts_config)

    def synthesize(self, text: str) -> "TtsResult":
        return self._t.synthesize(text)

    @property
    def sample_rate(self) -> int:
        return self.vocoder.sample_rate
