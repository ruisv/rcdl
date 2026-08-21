"""ByteTrack multi-object tracking tests.

These are PURE-NUMPY tests of the tracker — the oracle that ``rcdl::ByteTracker``
mirrors. They need only numpy and run anywhere: no board, no ``.rknn``, no
detector. The tracker consumes BOXES, not pixels, so its whole behaviour can be
stated with synthetic detections, which is the only way to write down what
SHOULD happen when motion and appearance disagree.

    PYTHONPATH=build:python pytest -s tests/test_tracking.py

The module-level ``RefByteTracker`` (plus ``ref_reid_preprocess`` and the
embedding primitives) is the documented "numpy path": an 8-D constant-velocity
Kalman filter, the padded Hungarian assignment, the two-stage association and
the track lifecycle, transcribed from src/tracks/byte_tracker.cc. Where the
compiled module exposes the same surface it is exercised against the oracle, but
the core assertions never depend on it.
"""

import math

import numpy as np
import pytest


# =========================================================================== #
# Embedding primitives — mirrors include/rcdl/tracks/reid.h                   #
# =========================================================================== #
def ref_l2_normalize(v):
    """L2-normalize to a unit vector; a zero vector is returned unchanged."""
    v = np.asarray(v, dtype=np.float32).copy()
    n = float(np.sqrt(np.sum(v.astype(np.float64) ** 2)))
    if n <= 0.0:
        return v
    return (v * np.float32(1.0 / n)).astype(np.float32)


def ref_cosine_similarity(a, b):
    """Normalized dot product in [-1, 1]; 0 on length mismatch or a zero side."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    na, nb = float(a @ a), float(b @ b)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float((a @ b) / (math.sqrt(na) * math.sqrt(nb)))


# =========================================================================== #
# ReID crop preprocessing — mirrors src/tracks/reid.cc                        #
# =========================================================================== #
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _lround(v):
    """C's lround: half away from zero (Python's round() is half-to-even)."""
    return int(math.copysign(math.floor(abs(v) + 0.5), v))


def ref_reid_preprocess(bgr, bx1, by1, bx2, by2, in_w=128, in_h=256,
                        mean=IMAGENET_MEAN, std=IMAGENET_STD, swap_rb=True):
    """Crop -> squashing bilinear resize -> RGB -> ImageNet z-score, NCHW.

    Deliberately NOT a letterbox: person-ReID models are trained on crops
    squashed to a fixed 2:1 shape, so preserving aspect would feed them padding
    bars they have never seen. Sampling uses the OpenCV pixel-center convention,
    which is what makes this agree with cv2.resize(INTER_LINEAR).
    """
    h, w = bgr.shape[:2]
    x1 = int(np.clip(_lround(bx1), 0, w))
    y1 = int(np.clip(_lround(by1), 0, h))
    x2 = int(np.clip(_lround(bx2), 0, w))
    y2 = int(np.clip(_lround(by2), 0, h))
    cw, ch = x2 - x1, y2 - y1
    if cw <= 0 or ch <= 0:
        raise ValueError("reid_preprocess: box is empty after clipping")

    crop = bgr[y1:y2, x1:x2].astype(np.float32)
    fy = np.clip((np.arange(in_h) + 0.5) * (ch / in_h) - 0.5, 0.0, ch - 1.0)
    fx = np.clip((np.arange(in_w) + 0.5) * (cw / in_w) - 0.5, 0.0, cw - 1.0)
    py0 = fy.astype(np.int32)
    px0 = fx.astype(np.int32)
    py1 = np.minimum(py0 + 1, ch - 1)
    px1 = np.minimum(px0 + 1, cw - 1)
    wy = (fy - py0)[:, None, None]
    wx = (fx - px0)[None, :, None]

    p00 = crop[np.ix_(py0, px0)]
    p01 = crop[np.ix_(py0, px1)]
    p10 = crop[np.ix_(py1, px0)]
    p11 = crop[np.ix_(py1, px1)]
    v = ((1 - wy) * (1 - wx) * p00 + (1 - wy) * wx * p01 +
         wy * (1 - wx) * p10 + wy * wx * p11)
    if swap_rb:
        v = v[..., ::-1]  # source is BGR, the model wants RGB
    v = v / 255.0
    v = (v - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(v.transpose(2, 0, 1)[None], dtype=np.float32)


# =========================================================================== #
# Reference ByteTracker — mirrors src/tracks/byte_tracker.cc                   #
# =========================================================================== #
STD_POS = 1.0 / 20.0   # _std_weight_position
STD_VEL = 1.0 / 160.0  # _std_weight_velocity

NEW, TRACKED, LOST, REMOVED = 0, 1, 2, 3

MIN_BOX_SIDE = 1e-3
"""Smallest side a tracklet may carry. The Kalman state is [cx, cy, w/h, h], so
a zero-height box makes the aspect term infinite and the tlwh round-trip yields
NaN, which then poisons a whole row of the assignment cost."""

MAHALANOBIS_LIMIT = 13.2767  # 4-dof chi-square 99% point


def _clamp_side(v):
    return v if v > MIN_BOX_SIDE else MIN_BOX_SIDE


def ref_box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    uni = ua + ub - inter
    return inter / uni if uni > 0.0 else 0.0


class RefKalman:
    """8-D constant-velocity box filter, state [cx, cy, a, h, vcx, vcy, va, vh].

    The measurement is xyah = (cx, cy, w/h, h). Process and observation noise
    scale with the box height (the reference's relative-uncertainty trick), so a
    small far-away box is filtered harder than a large near one.
    """

    def __init__(self):
        f = np.eye(8)
        f[:4, 4:] = np.eye(4)  # dt = 1: position picks up velocity
        self.f = f

    @staticmethod
    def initiate(meas):
        mean = np.concatenate([np.asarray(meas, float), np.zeros(4)])
        h = meas[3]
        std = np.array([2 * STD_POS * h, 2 * STD_POS * h, 1e-2, 2 * STD_POS * h,
                        10 * STD_VEL * h, 10 * STD_VEL * h, 1e-5, 10 * STD_VEL * h])
        return mean, np.diag(std ** 2)

    def predict(self, mean, cov):
        h = mean[3]
        std = np.array([STD_POS * h, STD_POS * h, 1e-2, STD_POS * h,
                        STD_VEL * h, STD_VEL * h, 1e-5, STD_VEL * h])
        return self.f @ mean, self.f @ cov @ self.f.T + np.diag(std ** 2)

    @staticmethod
    def project(mean, cov):
        h = mean[3]
        std = np.array([STD_POS * h, STD_POS * h, 1e-1, STD_POS * h])
        return mean[:4].copy(), cov[:4, :4] + np.diag(std ** 2)

    def update(self, mean, cov, meas):
        pmean, pcov = self.project(mean, cov)
        # gain[i] = S^-1 * cov[i, :4]  (the explicit form of scipy's cho_solve)
        gain = np.linalg.solve(pcov, cov[:, :4].T).T
        innov = np.asarray(meas, float) - pmean
        return mean + gain @ innov, cov - gain @ pcov @ gain.T


class RefSTrack:
    """One tracklet: Kalman state + lifecycle + appearance template."""

    def __init__(self, tlwh, score, class_id, feat=None):
        self.tlwh_init = np.array(
            [tlwh[0], tlwh[1], _clamp_side(tlwh[2]), _clamp_side(tlwh[3])], float)
        self.score = float(score)
        self.class_id = int(class_id)
        self.curr_feat = ref_l2_normalize(feat) if feat is not None and len(feat) else None
        self.smooth_feat = None
        self.mean = None
        self.cov = None
        self.track_id = 0
        self.tracklet_len = 0
        self.frame_id = 0
        self.start_frame = 0
        self.is_activated = False
        self.state = NEW

    # --- geometry --------------------------------------------------------
    def tlwh(self):
        if self.mean is None:
            return self.tlwh_init.copy()
        w = self.mean[2] * self.mean[3]  # a * h
        h = self.mean[3]
        return np.array([self.mean[0] - w / 2.0, self.mean[1] - h / 2.0, w, h])

    def tlbr(self):
        t = self.tlwh()
        return (t[0], t[1], t[0] + t[2], t[1] + t[3])

    @staticmethod
    def tlwh_to_xyah(t):
        h = _clamp_side(t[3])
        return np.array([t[0] + t[2] / 2.0, t[1] + h / 2.0, _clamp_side(t[2]) / h, h])

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh())

    # --- lifecycle -------------------------------------------------------
    def predict(self, kf):
        # Zero the height-velocity of a non-tracked tracklet before predicting
        # (the reference's multi_predict trick): a coasting box should not keep
        # growing or shrinking on nothing but its last measured trend.
        if self.state != TRACKED:
            self.mean[7] = 0.0
        self.mean, self.cov = kf.predict(self.mean, self.cov)

    def activate(self, kf, frame_id, new_id):
        self.track_id = new_id
        self.mean, self.cov = kf.initiate(self.tlwh_to_xyah(self.tlwh_init))
        self.tracklet_len = 0
        self.state = TRACKED
        self.is_activated = (frame_id == 1)  # only frame-1 starts confirmed
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.smooth_feat = self.curr_feat  # this tracklet IS the spawning detection

    def re_activate(self, kf, det, frame_id, alpha):
        self.mean, self.cov = kf.update(self.mean, self.cov,
                                        self.tlwh_to_xyah(det.tlwh()))
        self.update_features(det.curr_feat, alpha)
        self.tracklet_len = 0
        self.state = TRACKED
        self.is_activated = True
        self.frame_id = frame_id
        self.score = det.score
        self.class_id = det.class_id

    def update(self, kf, det, frame_id, alpha):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.cov = kf.update(self.mean, self.cov,
                                        self.tlwh_to_xyah(det.tlwh()))
        self.update_features(det.curr_feat, alpha)
        self.state = TRACKED
        self.is_activated = True
        self.score = det.score
        self.class_id = det.class_id

    # --- appearance ------------------------------------------------------
    def update_features(self, feat, alpha):
        """Fold a detection's (unit-norm) embedding into this tracklet's
        template and re-normalize, so cosine against it stays a dot product."""
        if feat is None or len(feat) == 0:
            return
        self.curr_feat = feat
        if self.smooth_feat is None or len(self.smooth_feat) != len(feat):
            self.smooth_feat = feat  # first sighting
            return
        self.smooth_feat = ref_l2_normalize(
            alpha * np.asarray(self.smooth_feat, np.float64) +
            (1.0 - alpha) * np.asarray(feat, np.float64))

    # --- BoostTrack++ support --------------------------------------------
    def confidence(self, frame_id):
        """How much this tracklet's PREDICTED box deserves to be trusted."""
        coef, young = 0.9, 7
        age = frame_id - self.start_frame
        if age < young:
            return coef ** (young - age)
        return coef ** (self.since_update(frame_id) - 1)

    def since_update(self, frame_id):
        return max(1, frame_id - self.frame_id)

    def mahalanobis_sq(self, meas):
        """Squared Mahalanobis distance using the DIAGONAL of the covariance —
        the reference's approximation, kept because a full solve would cost a
        Cholesky per pair."""
        if self.mean is None:
            return float("inf")
        acc = 0.0
        for i in range(4):
            var = self.cov[i, i]
            if var <= 1e-12:
                return float("inf")
            d = meas[i] - self.mean[i]
            acc += d * d / var
        return acc if math.isfinite(acc) else float("inf")

    def camera_update(self, m):
        """Warp the box by a 2x3 affine, leaving the velocities alone."""
        if self.mean is None:
            return
        b = self.tlbr()
        x1 = m[0] * b[0] + m[1] * b[1] + m[2]
        y1 = m[3] * b[0] + m[4] * b[1] + m[5]
        x2 = m[0] * b[2] + m[1] * b[3] + m[2]
        y2 = m[3] * b[2] + m[4] * b[3] + m[5]
        w, h = x2 - x1, y2 - y1
        if w <= 0.0 or h <= 0.0:
            return  # degenerate warp: leave the track alone
        self.mean[0] = x1 + w / 2.0
        self.mean[1] = y1 + h / 2.0
        self.mean[2] = w / h
        self.mean[3] = h


# --------------------------------------------------------------------------- #
# Association: IoU cost, score fusion, appearance fusion, assignment           #
# --------------------------------------------------------------------------- #
def ref_iou_distance(tracks, dets):
    """cost[i][j] = 1 - IoU(track_i, det_j)."""
    r, c = len(tracks), len(dets)
    cost = np.zeros((r, c))
    tb = [t.tlbr() for t in tracks]
    db = [d.tlbr() for d in dets]
    for i in range(r):
        for j in range(c):
            cost[i, j] = 1.0 - ref_box_iou(tb[i], db[j])
    return cost


def ref_fuse_score(cost, dets):
    """dist = 1 - (1 - dist) * det.score — a confident detection is cheaper."""
    if cost.size == 0:
        return cost
    s = np.array([d.score for d in dets])
    return 1.0 - (1.0 - cost) * s[None, :]


def ref_fuse_appearance(cost, raw_iou, tracks, dets, cfg):
    """BoT-SORT appearance fusion: cost = min(iou_cost, gated_cosine_dist).

    Taking the MINIMUM is what makes this safe — appearance can only lower a
    cost, so a bad embedding can never break a match that geometry already had.
    A pair is excluded (left at its geometric cost) when either side has no
    embedding, the classes differ (there is no shared appearance space between
    classes), the boxes are too far apart, or the vectors are too dissimilar.
    """
    for i, t in enumerate(tracks):
        tf = t.smooth_feat
        if tf is None or len(tf) == 0:
            continue
        for j, d in enumerate(dets):
            df = d.curr_feat
            if df is None or len(df) != len(tf):
                continue
            if t.class_id != d.class_id:
                continue
            if raw_iou[i, j] > cfg.proximity_thresh:
                continue
            # Both are unit norm, so cosine is a dot product. Halve the distance
            # to the reference's scale, on which appearance_thresh is set.
            emb_dist = (1.0 - float(np.dot(np.asarray(tf, np.float64),
                                           np.asarray(df, np.float64)))) / 2.0
            if emb_dist > cfg.appearance_thresh:
                continue
            cost[i, j] = min(cost[i, j], emb_dist)
    return cost


def ref_linear_assignment(cost, thresh):
    """Minimum-cost assignment with a gate, mirroring lapjv(cost_limit=thresh).

    Pads the matrix to square, runs the O(n^3) potentials-based Hungarian, then
    keeps only real (row, col) pairs whose cost is <= ``thresh``. Detection
    counts are tens, so the cubic solve is free.

    Returns (matches, unmatched_rows, unmatched_cols).
    """
    r, c = cost.shape
    if r == 0 or c == 0:
        return [], list(range(r)), list(range(c))

    n = max(r, c)
    big, inf = 1e9, float("inf")
    a = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            a[i][j] = float(cost[i - 1, j - 1]) if (i <= r and j <= c) else big

    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)     # p[j] = row matched to column j
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = inf, -1
            for j in range(1, n + 1):
                if used[j]:
                    continue
                if j1 < 0:
                    j1 = j  # first free column, as a fallback target
                cur = a[i0][j] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 < 0:
                break  # every column used: cannot happen while row i is free
            if not delta < inf:
                delta = 0.0  # all-NaN row; the gate below rejects the pair anyway
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    matches = []
    row_used = [False] * r
    col_used = [False] * c
    for j in range(1, n + 1):
        i = p[j]
        if 1 <= i <= r and j <= c and cost[i - 1, j - 1] <= thresh:
            matches.append((i - 1, j - 1))
            row_used[i - 1] = True
            col_used[j - 1] = True
    return (matches,
            [i for i in range(r) if not row_used[i]],
            [j for j in range(c) if not col_used[j]])


# --------------------------------------------------------------------------- #
# BoostTrack++ similarity terms                                                #
# --------------------------------------------------------------------------- #
def ref_soft_biou(det, trk, track_conf):
    """Grow both boxes in proportion to how little the tracklet is trusted, then
    take the IoU: a tracklet that has been coasting searches a wider area."""
    k_det, k_trk = 0.25, 0.5
    slack = 1.0 - track_conf
    dx = (det[2] - det[0]) * slack * k_det
    dy = (det[3] - det[1]) * slack * k_det
    tx = (trk[2] - trk[0]) * slack * k_trk
    ty = (trk[3] - trk[1]) * slack * k_trk
    a = (det[0] - dx, det[1] - dy, det[2] + dx, det[3] + dy)
    b = (trk[0] - tx, trk[1] - ty, trk[2] + tx, trk[3] + ty)
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uni if uni > 0.0 else 0.0


def ref_shape_similarity(det, trk):
    """exp of the negative relative width and height mismatch, in [0,1]."""
    dw, dh = det[2] - det[0], det[3] - det[1]
    tw, th = trk[2] - trk[0], trk[3] - trk[1]
    mw, mh = max(dw, tw), max(dh, th)
    if mw <= 0.0 or mh <= 0.0:
        return 0.0
    return math.exp(-(abs(dw - tw) / mw + abs(dh - th) / mh))


def ref_mahalanobis_similarity(dist):
    """(n_det, n_track) squared distances -> similarities.

    Clamp at the chi-square limit, invert, softmax DOWN EACH TRACK's column, and
    zero out everything that was clamped. The softmax makes this a competition
    between detections for one tracklet rather than an absolute score.
    """
    sim = np.zeros_like(dist)
    if dist.size == 0:
        return sim
    ok = dist <= MAHALANOBIS_LIMIT  # NaN-safe: also excludes non-finite entries
    e = np.where(ok, np.exp(MAHALANOBIS_LIMIT - np.where(ok, dist, 0.0)), 0.0)
    s = e.sum(axis=0)
    good = s > 0.0
    sim[:, good] = e[:, good] / s[good]
    return sim


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
class RefBoostConfig:
    """All off by default — `{}` must mean plain ByteTrack, or the ablation is
    meaningless."""

    def __init__(self):
        self.rich_similarity = False
        self.soft_biou = False
        self.boost_detections = False
        self.lambda_iou = 0.5
        self.lambda_mhd = 0.25
        self.lambda_shape = 0.25
        self.min_iou = 0.3
        self.dlo_alpha = 0.65
        self.vt_start = 0.95
        self.vt_end = 0.80
        self.vt_steps = 20
        self.duo = True
        self.duo_iou = 0.3


class RefByteTrackConfig:
    def __init__(self, **kw):
        self.track_thresh = 0.5
        self.high_thresh = 0.6
        self.match_thresh = 0.8
        self.track_buffer = 30
        self.frame_rate = 30
        self.proximity_thresh = 0.5
        self.appearance_thresh = 0.25
        self.ema_alpha = 0.95
        self.boost = RefBoostConfig()
        for k, val in kw.items():
            setattr(self, k, val)


class RefTrack:
    """One tracked object for the current frame, in original-image pixels."""

    def __init__(self, track_id, box, score, class_id):
        self.track_id = track_id
        self.x1, self.y1, self.x2, self.y2 = box
        self.score = score
        self.class_id = class_id

    def __repr__(self):
        return (f"Track(id={self.track_id} cls={self.class_id} score={self.score:.3f} "
                f"box=[{self.x1:.1f},{self.y1:.1f},{self.x2:.1f},{self.y2:.1f}])")


# --------------------------------------------------------------------------- #
# The tracker                                                                  #
# --------------------------------------------------------------------------- #
def _joint(a, b):
    res, seen = list(a), {t.track_id for t in a}
    for t in b:
        if t.track_id not in seen:
            seen.add(t.track_id)
            res.append(t)
    return res


def _sub(a, b):
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]


def _remove_duplicates(a, b):
    """Drop near-duplicate tracks across two lists (IoU distance < 0.15),
    keeping the longer-lived one."""
    if not a or not b:
        return a, b
    cost = ref_iou_distance(a, b)
    dupa = [False] * len(a)
    dupb = [False] * len(b)
    for p in range(len(a)):
        for q in range(len(b)):
            if cost[p, q] >= 0.15:
                continue
            tp = a[p].frame_id - a[p].start_frame
            tq = b[q].frame_id - b[q].start_frame
            if tp > tq:
                dupb[q] = True
            else:
                dupa[p] = True
    return ([t for i, t in enumerate(a) if not dupa[i]],
            [t for j, t in enumerate(b) if not dupb[j]])


class RefByteTracker:
    """Stateful ByteTrack, transcribed from src/tracks/byte_tracker.cc.

    The per-frame pipeline, in order:
      1. Kalman-predict every tracked and recently-lost tracklet.
      2. (optional) BoostTrack detection boosting, BEFORE the high/low split.
      3. First association: high-score detections vs the whole pool, IoU cost
         fused with the detection score (and appearance when embeddings are on).
      4. Second association: the leftover STILL-TRACKED tracklets vs the LOW
         score detections. This is the "BYTE" idea and the whole reason weak
         boxes are kept instead of discarded.
      5. Unconfirmed tracklets vs the leftover high-score detections.
      6. Spawn new tracklets from what is still left above `high_thresh`.
      7. Expire lost tracklets older than `max_time_lost`.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else RefByteTrackConfig()
        if self.cfg.frame_rate <= 0:
            raise ValueError("ByteTracker: frame_rate must be > 0")
        if not 0.0 <= self.cfg.ema_alpha <= 1.0:
            raise ValueError("ByteTracker: ema_alpha must be in [0, 1]")
        self.kf = RefKalman()
        self.tracked = []
        self.lost = []
        self.frame_id = 0
        self.id_count = 0
        self.embed_dim = 0
        self.max_time_lost = int(round(self.cfg.frame_rate / 30.0 * self.cfg.track_buffer))

    def reset(self):
        self.tracked, self.lost = [], []
        self.frame_id = 0
        self.id_count = 0
        self.embed_dim = 0

    def apply_camera_motion(self, affine):
        m = np.asarray(affine, float).reshape(-1)
        for t in self.tracked:
            t.camera_update(m)
        for t in self.lost:
            t.camera_update(m)

    def _next_id(self):
        self.id_count += 1
        return self.id_count

    def _ema_alpha(self, score):
        """Scale from `ema_alpha` at full confidence up to 1.0 (no update at
        all) as the score approaches the high/low split, so the crops least
        likely to show the whole target contribute least to the template."""
        span = 1.0 - self.cfg.track_thresh
        trust = 1.0 if span <= 1e-6 else (score - self.cfg.track_thresh) / span
        trust = min(max(trust, 0.0), 1.0)
        a = self.cfg.ema_alpha
        return a + (1.0 - a) * (1.0 - trust)

    # --- BoostTrack++ ----------------------------------------------------
    def _combined_similarity(self, det_boxes, det_xyah, pool):
        """Mean of soft buffered IoU, Mahalanobis similarity and shape
        agreement — the CONSENSUS of three notions of "same object"."""
        r, c = len(det_boxes), len(pool)
        mh = np.array([[pool[j].mahalanobis_sq(det_xyah[i]) for j in range(c)]
                       for i in range(r)]) if r and c else np.zeros((r, c))
        mh_sim = ref_mahalanobis_similarity(mh)
        s = np.zeros((r, c))
        for j in range(c):
            conf = pool[j].confidence(self.frame_id)
            tb = pool[j].tlbr()
            for i in range(r):
                s[i, j] = (mh_sim[i, j] + ref_shape_similarity(det_boxes[i], tb) +
                           ref_soft_biou(det_boxes[i], tb, conf)) / 3.0
        return s

    def _boost_detections(self, dets, pool, scores):
        r, c = len(dets), len(pool)
        if r == 0 or c == 0:
            return
        det_boxes = [(d["x1"], d["y1"], d["x2"], d["y2"]) for d in dets]
        det_xyah = [RefSTrack.tlwh_to_xyah((d["x1"], d["y1"],
                                            d["x2"] - d["x1"], d["y2"] - d["y1"]))
                    for d in dets]
        s = self._combined_similarity(det_boxes, det_xyah, pool)
        b = self.cfg.boost
        for i in range(r):
            best, vouched = 0.0, False
            for j in range(c):
                sij = s[i, j]
                best = max(best, sij)
                # The longer a tracklet has gone unmatched the more its box has
                # drifted, so demand less similarity from it.
                step = (b.vt_start - b.vt_end) / max(1, b.vt_steps)
                thr = max(b.vt_start - (pool[j].since_update(self.frame_id) - 1) * step,
                          b.vt_end)
                if sij > thr:
                    vouched = True
            soft = b.dlo_alpha * scores[i] + (1.0 - b.dlo_alpha) * best ** 1.5
            scores[i] = max(scores[i], soft)
            if vouched:
                scores[i] = max(scores[i], self.cfg.high_thresh + 1e-5)

        if not b.duo:
            return
        # DUO: a weak detection that NO tracklet can explain is more likely a
        # new object than a false positive. "Cannot explain" is the Mahalanobis
        # limit, not IoU — a tracklet may overlap a box and still be a poor
        # motion explanation for it.
        cand = []
        for i in range(r):
            if scores[i] >= self.cfg.high_thresh:
                continue
            min_mh = min(pool[j].mahalanobis_sq(det_xyah[i]) for j in range(c))
            if min_mh > MAHALANOBIS_LIMIT:
                cand.append(i)
        for i in cand:
            best_of_cluster = True
            for k in cand:
                if k == i:
                    continue
                if (ref_box_iou(det_boxes[i], det_boxes[k]) > b.duo_iou and
                        (scores[k] > scores[i] or (scores[k] == scores[i] and k < i))):
                    best_of_cluster = False
                    break
            if best_of_cluster:
                scores[i] = max(scores[i], self.cfg.high_thresh + 1e-4)

    def _add_boost_bonuses(self, cost, raw_iou, tracks, dets):
        b = self.cfg.boost
        r, c = len(tracks), len(dets)
        if r == 0 or c == 0:
            return
        det_boxes = [d.tlbr() for d in dets]
        det_xyah = [d.to_xyah() for d in dets]
        # Mahalanobis similarity is normalized down each tracklet's column, so
        # it has to be built over the whole matrix rather than per pair.
        mh = np.array([[tracks[i].mahalanobis_sq(det_xyah[j]) for i in range(r)]
                       for j in range(c)])
        mh_sim = ref_mahalanobis_similarity(mh)
        for i in range(r):
            tconf = tracks[i].confidence(self.frame_id)
            tb = tracks[i].tlbr()
            for j in range(c):
                iou = 1.0 - raw_iou[i, j]
                # The confidence product gates the IoU and shape bonuses: a pair
                # that is not already plausible in space gets no bonus at all.
                conf = 0.0 if iou < b.min_iou else tconf * dets[j].score
                bonus = b.lambda_mhd * mh_sim[j, i]
                bonus += b.lambda_iou * conf * iou
                bonus += b.lambda_shape * conf * ref_shape_similarity(det_boxes[j], tb)
                cost[i, j] -= bonus

    # --- the per-frame update -------------------------------------------
    def update(self, dets, embeddings=None):
        self.frame_id += 1
        cfg = self.cfg

        has_emb = embeddings is not None and len(embeddings) > 0
        if has_emb:
            if len(embeddings) != len(dets):
                raise ValueError("ByteTracker::update: embeddings must run parallel "
                                 "to detections (one entry each, empty entries allowed)")
            for e in embeddings:
                if e is None or len(e) == 0:
                    continue
                if self.embed_dim == 0:
                    self.embed_dim = len(e)
                if len(e) != self.embed_dim:
                    raise ValueError("ByteTracker::update: embedding width changed "
                                     "mid-stream (all ReID vectors must come from "
                                     "one model)")

        activated, refind, lost_now, removed_now = [], [], [], []

        unconfirmed = [t for t in self.tracked if not t.is_activated]
        tracked_stracks = [t for t in self.tracked if t.is_activated]

        # The Kalman predict happens BEFORE the high/low split, not after:
        # detection boosting compares detections against the tracklets'
        # PREDICTED boxes, and its whole purpose is to change which side of the
        # split a detection lands on. With boosting off this is the same work in
        # a different order.
        pool = _joint(tracked_stracks, self.lost)
        for t in pool:
            t.predict(self.kf)

        scores = [float(d["score"]) for d in dets]
        if cfg.boost.boost_detections:
            self._boost_detections(dets, pool, scores)

        def make_strack(i):
            d = dets[i]
            feat = embeddings[i] if has_emb else None
            return RefSTrack((d["x1"], d["y1"], d["x2"] - d["x1"], d["y2"] - d["y1"]),
                             scores[i], d.get("class_id", 0), feat)

        detections = [make_strack(i) for i in range(len(dets))
                      if scores[i] > cfg.track_thresh]
        detections_second = [make_strack(i) for i in range(len(dets))
                             if not scores[i] > cfg.track_thresh and scores[i] > 0.1]

        # --- Step 2: first association (high score) -----------------------
        raw_iou = ref_iou_distance(pool, detections)
        cost = raw_iou.copy()
        if cfg.boost.soft_biou:
            for i, t in enumerate(pool):
                conf, tb = t.confidence(self.frame_id), t.tlbr()
                for j, d in enumerate(detections):
                    cost[i, j] = 1.0 - ref_soft_biou(d.tlbr(), tb, conf)
        cost = ref_fuse_score(cost, detections)
        if has_emb:
            cost = ref_fuse_appearance(cost, raw_iou, pool, detections, cfg)
        if cfg.boost.rich_similarity:
            self._add_boost_bonuses(cost, raw_iou, pool, detections)
        matches, u_rows, u_cols = ref_linear_assignment(cost, cfg.match_thresh)
        if cfg.boost.rich_similarity:
            # The bonuses only ever lower a cost, so on their own they could
            # drag a geometrically impossible pair under the gate. Re-check the
            # RAW IoU of everything that matched.
            kept = [m for m in matches
                    if 1.0 - raw_iou[m[0], m[1]] >= cfg.boost.min_iou]
            rows = {m[0] for m in kept}
            cols = {m[1] for m in kept}
            matches = kept
            u_rows = [i for i in range(len(pool)) if i not in rows]
            u_cols = [j for j in range(len(detections)) if j not in cols]

        for i, j in matches:
            track, det = pool[i], detections[j]
            alpha = self._ema_alpha(det.score)
            if track.state == TRACKED:
                track.update(self.kf, det, self.frame_id, alpha)
                activated.append(track)
            else:
                track.re_activate(self.kf, det, self.frame_id, alpha)
                refind.append(track)

        # --- Step 3: second association (low score), still-tracked rows only
        # Pure IoU by design — these are the weak boxes, whose crops are exactly
        # the occluded / blurred ones.
        r_tracked = [pool[i] for i in u_rows if pool[i].state == TRACKED]
        cost2 = ref_iou_distance(r_tracked, detections_second)
        m2, u_rows2, _ = ref_linear_assignment(cost2, 0.5)
        for i, j in m2:
            track, det = r_tracked[i], detections_second[j]
            alpha = self._ema_alpha(det.score)
            if track.state == TRACKED:
                track.update(self.kf, det, self.frame_id, alpha)
                activated.append(track)
            else:
                track.re_activate(self.kf, det, self.frame_id, alpha)
                refind.append(track)
        for i in u_rows2:
            track = r_tracked[i]
            if track.state != LOST:
                track.state = LOST
                lost_now.append(track)

        # --- unconfirmed tracks vs leftover high-score detections ---------
        det_left = [detections[j] for j in u_cols]
        raw_iou3 = ref_iou_distance(unconfirmed, det_left)
        cost3 = ref_fuse_score(raw_iou3.copy(), det_left)
        if has_emb:
            cost3 = ref_fuse_appearance(cost3, raw_iou3, unconfirmed, det_left, cfg)
        m3, u_rows3, u_cols3 = ref_linear_assignment(cost3, 0.7)
        for i, j in m3:
            det = det_left[j]
            unconfirmed[i].update(self.kf, det, self.frame_id, self._ema_alpha(det.score))
            activated.append(unconfirmed[i])
        for i in u_rows3:
            unconfirmed[i].state = REMOVED
            removed_now.append(unconfirmed[i])

        # --- Step 4: init new tracks from leftover high-score detections ---
        for j in u_cols3:
            track = det_left[j]
            if track.score < cfg.high_thresh:
                continue
            track.activate(self.kf, self.frame_id, self._next_id())
            activated.append(track)

        # --- Step 5: expire lost tracks older than the recovery buffer -----
        for track in self.lost:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = REMOVED
                removed_now.append(track)

        # --- merge / dedup the persistent lists ----------------------------
        new_tracked = [t for t in self.tracked if t.state == TRACKED]
        new_tracked = _joint(new_tracked, activated)
        new_tracked = _joint(new_tracked, refind)

        self.lost = _sub(self.lost, new_tracked) + lost_now
        self.lost = _sub(self.lost, removed_now)
        new_tracked, self.lost = _remove_duplicates(new_tracked, self.lost)
        self.tracked = new_tracked

        return [RefTrack(t.track_id, t.tlbr(), t.score, t.class_id)
                for t in self.tracked if t.is_activated]


# =========================================================================== #
# Scene helpers                                                               #
# =========================================================================== #
def det(x, y=0.0, w=60.0, h=120.0, score=0.9, cls=0):
    return {"x1": float(x), "y1": float(y), "x2": float(x + w), "y2": float(y + h),
            "score": float(score), "class_id": int(cls)}


def ids_left_to_right(tracks):
    """Track ids ordered by box position, left first.

    Deliberately ORDER rather than absolute position: a matched track's box is a
    Kalman estimate that sits partway between prediction and measurement, so
    asserting on exact pixels would be asserting on the filter's gain.
    """
    return [t.track_id for t in sorted(tracks, key=lambda t: t.x1)]


def longest_id_run(per_frame_ids):
    """Longest run of consecutive frames in which some single id appears."""
    best, active = 0, {}
    for ids in per_frame_ids:
        nxt = {}
        for i in ids:
            nxt[i] = active.get(i, 0) + 1
            best = max(best, nxt[i])
        active = nxt
    return best


# =========================================================================== #
# Optional C++ cross-check                                                    #
# =========================================================================== #
@pytest.fixture(scope="module")
def cxx():
    """The compiled module, or None — every test still asserts on the oracle."""
    try:
        import rcdl
    except Exception:
        return None
    return rcdl if hasattr(rcdl, "ByteTracker") else None


def _cxx_det(mod, d):
    o = mod.Detection()
    o.x1, o.y1, o.x2, o.y2 = d["x1"], d["y1"], d["x2"], d["y2"]
    o.score, o.class_id = d["score"], d["class_id"]
    return o


def _cxx_run(mod, frames, **cfg_kw):
    """Feed the same frames to the compiled tracker; returns per-frame id sets."""
    cfg = mod.ByteTrackConfig()
    for k, val in cfg_kw.items():
        setattr(cfg, k, val)
    t = mod.ByteTracker(cfg)
    return [sorted(x.track_id for x in t.update([_cxx_det(mod, d) for d in f]))
            for f in frames]


def _ref_run(frames, **cfg_kw):
    t = RefByteTracker(RefByteTrackConfig(**cfg_kw))
    return [sorted(x.track_id for x in t.update(f)) for f in frames]


# =========================================================================== #
# Embedding primitives                                                        #
# =========================================================================== #
def test_l2_normalize_unit_norm():
    out = ref_l2_normalize([3.0, 4.0])  # norm 5
    assert out == pytest.approx([0.6, 0.8], abs=1e-5)
    assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-5)


def test_l2_normalize_zero_vector_is_a_noop():
    assert ref_l2_normalize(np.zeros(4, np.float32)) == pytest.approx([0, 0, 0, 0])


def test_cosine_similarity():
    a = [1.0, 0.0, 0.0]
    assert ref_cosine_similarity(a, [1.0, 0.0, 0.0]) == pytest.approx(1.0, abs=1e-6)
    assert ref_cosine_similarity(a, [0.0, 1.0, 0.0]) == pytest.approx(0.0, abs=1e-6)
    assert ref_cosine_similarity(a, [-1.0, 0.0, 0.0]) == pytest.approx(-1.0, abs=1e-6)
    # unnormalized inputs are normalized internally
    assert ref_cosine_similarity([2.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_length_mismatch_is_zero():
    assert ref_cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_embedding_primitives_match_cxx(cxx):
    if cxx is None or not hasattr(cxx, "normalize_embedding"):
        pytest.skip("compiled rcdl module without reid primitives")
    v = np.array([3.0, 4.0, 12.0], np.float32)
    got = np.asarray(cxx.normalize_embedding(v), np.float32)
    assert got == pytest.approx(ref_l2_normalize(v), abs=1e-5)
    assert cxx.cosine_similarity([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(
        ref_cosine_similarity([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]), abs=1e-5)


# =========================================================================== #
# The four behaviours a tracker exists to provide                             #
# =========================================================================== #
def test_straight_line_track_keeps_one_id():
    """A constant-velocity object must hold ONE id for its whole life.

    Nothing about the geometry is ambiguous here, so any id churn is the Kalman
    prediction or the association gate misbehaving rather than a hard call.
    """
    t = RefByteTracker()
    seen, per_frame = set(), []
    for f in range(40):
        tracks = t.update([det(100 + 12 * f, y=50)])
        per_frame.append({x.track_id for x in tracks})
        seen.update(x.track_id for x in tracks)
    assert seen == {1}, f"one object produced {len(seen)} ids: {sorted(seen)}"
    # Confirmed from frame 1 onwards (a track created on frame 1 starts
    # confirmed; one created later needs a second sighting).
    assert longest_id_run(per_frame) == 40


def test_track_box_follows_the_object():
    """The emitted box is a Kalman estimate, but it must track the measurement
    rather than lag it indefinitely."""
    t = RefByteTracker()
    for f in range(20):
        tracks = t.update([det(100 + 12 * f, y=50)])
    assert len(tracks) == 1
    assert tracks[0].x1 == pytest.approx(100 + 12 * 19, abs=3.0)
    assert tracks[0].y1 == pytest.approx(50.0, abs=3.0)
    assert tracks[0].x2 - tracks[0].x1 == pytest.approx(60.0, abs=3.0)


@pytest.mark.parametrize("gap,same_id", [(1, True), (5, True), (6, False), (12, False)])
def test_lost_track_expires_after_track_buffer(gap, same_id):
    """A track that disappears is kept for `track_buffer` frames and then gone.

    Within the buffer the SAME id is handed back (that is the recovery buffer's
    entire purpose); past it the object is a stranger and must get a NEW id —
    silently reusing the id would claim an identity the tracker cannot support.

    With track_buffer=5 the tracklet last seen on frame 5 is expired on the
    first frame F with F - 5 > 5, i.e. F = 11, so a gap of 5 empty frames still
    re-finds it and a gap of 6 cannot.
    """
    cfg = RefByteTrackConfig(track_buffer=5)
    t = RefByteTracker(cfg)
    for _ in range(5):
        first = [x.track_id for x in t.update([det(200, y=100)])]
    assert first == [1]

    for _ in range(gap):
        assert t.update([]) == [], "a vanished object must not be emitted"

    t.update([det(200, y=100)])       # re-created tracks start unconfirmed...
    again = [x.track_id for x in t.update([det(200, y=100)])]  # ...confirmed here
    assert len(again) == 1
    if same_id:
        assert again == first, "within the buffer the id must be recovered"
    else:
        assert again != first, "past the buffer the object must get a NEW id"
        assert again == [2]


def test_crossing_tracks_do_not_swap_ids():
    """Two objects moving through each other must keep their own ids.

    At the crossing frame the boxes overlap almost perfectly, so IoU alone is a
    coin flip; what decides it is the constant-velocity PREDICTION, which puts
    each tracklet on the far side of the other before the association runs. This
    is the case that a tracker without a motion model gets wrong.
    """
    t = RefByteTracker()
    left_id = right_id = None
    per_frame = []
    for f in range(21):
        a = 100 + 20 * f     # moving right
        b = 500 - 20 * f     # moving left
        tracks = t.update([det(a, y=80), det(b, y=80)])
        per_frame.append(tracks)
        if f == 0:
            assert len(tracks) == 2
            left_id, right_id = ids_left_to_right(tracks)

    # After the crossing the ids have exchanged SIDES but not identities: the
    # one that started on the left is now on the right.
    final = ids_left_to_right(per_frame[-1])
    assert len(final) == 2
    assert final == [right_id, left_id], (
        f"ids swapped at the crossing: started {[left_id, right_id]}, ended {final}")
    # And no third id was ever invented along the way.
    all_ids = {x.track_id for fr in per_frame for x in fr}
    assert all_ids == {left_id, right_id}, f"crossing spawned extra tracks: {all_ids}"


def test_low_score_detections_rescue_a_track():
    """The second association is the whole point of ByteTrack.

    A partially occluded object comes back as a WEAK detection. Kept in the low
    pool (score in (0.1, track_thresh]) it still re-anchors its tracklet; dropped
    entirely (score <= 0.1) the tracklet coasts, is marked lost and stops being
    emitted. Same frames, same boxes — only which pool the box lands in differs.
    """
    def run(weak_score):
        t = RefByteTracker()
        for f in range(5):
            strong = [x.track_id for x in t.update([det(100 + 10 * f, y=60)])]
        weak_frames = []
        for f in range(5, 11):
            weak_frames.append([x.track_id for x in
                                t.update([det(100 + 10 * f, y=60, score=weak_score)])])
        return strong, weak_frames

    strong, rescued = run(0.35)   # low pool: above 0.1, below track_thresh 0.5
    assert strong == [1]
    assert all(f == [1] for f in rescued), (
        f"the low-score pass should have held the track: {rescued}")

    strong, dropped = run(0.05)   # below 0.1: discarded before association
    assert strong == [1]
    assert dropped[0] == [], "a discarded detection cannot hold the track"
    assert all(f == [] for f in dropped)


def test_low_score_detections_cannot_start_a_track():
    """The weak pool rescues tracks; it never creates them. Otherwise every
    flicker of detector noise would become an identity."""
    t = RefByteTracker()
    for _ in range(6):
        tracks = t.update([det(300, y=30, score=0.35)])
    assert tracks == []


# =========================================================================== #
# Appearance-gated association (BoT-SORT)                                     #
# =========================================================================== #
#
# The scenario below is built so that geometry and appearance give DIFFERENT
# answers, because that is the only case where appearance can be shown to do
# anything. Two same-size boxes sit 25 px apart with 100 px width, so they
# overlap at IoU 0.6 — close enough that every cross pair passes the proximity
# gate (raw IoU distance 0.4 <= proximity_thresh 0.5) and appearance is allowed
# to speak at all. Then the two objects trade places in one frame. Motion says
# "each id keeps its position"; appearance says "each id follows its vector".

RED = [1.0, 0.0, 0.0, 0.0]
BLUE = [0.0, 1.0, 0.0, 0.0]


def _run_swap(use_appearance, cfg=None, final_embs=None):
    """Two overlapping objects hold still, then trade appearance on the last
    frame. Returns (ids_before, ids_after) keyed by position."""
    t = RefByteTracker(cfg)
    before = None
    for _ in range(4):  # settle: both tracks confirmed, velocities ~0
        dets = [det(100, w=100.0, h=200.0), det(125, w=100.0, h=200.0)]
        before = ids_left_to_right(
            t.update(dets, [RED, BLUE]) if use_appearance else t.update(dets))

    dets = [det(100, w=100.0, h=200.0), det(125, w=100.0, h=200.0)]
    embs = final_embs if final_embs is not None else [BLUE, RED]
    after = ids_left_to_right(t.update(dets, embs) if use_appearance else t.update(dets))
    return before, after


def _drifted(base, cos_to_base):
    """A unit vector at a chosen cosine from `base` (which must be an axis)."""
    v = np.array(base, np.float64) * cos_to_base
    v[3] = math.sqrt(max(0.0, 1.0 - cos_to_base ** 2))  # spare axis
    return [float(x) for x in v]


def test_motion_only_ids_follow_position():
    """Baseline: without embeddings the ids stay pinned to the boxes."""
    before, after = _run_swap(use_appearance=False)
    assert len(before) == 2
    assert after == before, "motion alone keeps each id on its side"


def test_appearance_decides_an_ambiguous_pairing():
    """With embeddings the ids follow the appearance instead of the position —
    when two candidates are both geometrically plausible, the one that LOOKS
    right wins."""
    before, after = _run_swap(use_appearance=True)
    assert len(before) == 2
    assert after == before[::-1], "each id should have crossed to its own colour"


def test_proximity_gate_blocks_a_distant_appearance_match():
    """Appearance must not reach across space: tightening the proximity gate
    below the pair's IoU distance (0.4) puts the ids back on the boxes."""
    cfg = RefByteTrackConfig(proximity_thresh=0.1)
    before, after = _run_swap(use_appearance=True, cfg=cfg)
    assert after == before


def test_appearance_gate_blocks_a_dissimilar_match():
    """The cosine gate must actually gate. The swapped objects come back with
    their appearance DRIFTED to cosine 0.9 (distance (1-0.9)/2 = 0.05), which the
    default 0.25 accepts and a 0.01 threshold rejects — so the same frames give
    opposite answers on either side of the gate."""
    drifted = [_drifted(BLUE, 0.9), _drifted(RED, 0.9)]
    _, loose = _run_swap(use_appearance=True, final_embs=drifted)
    cfg = RefByteTrackConfig(appearance_thresh=0.01)
    before, tight = _run_swap(use_appearance=True, cfg=cfg, final_embs=drifted)
    assert loose == before[::-1], "0.05 < 0.25: drifted appearance still matches"
    assert tight == before, "0.05 > 0.01: gated out, geometry decides"


def test_class_gate_blocks_cross_class_appearance():
    """Appearance is never compared across classes — there is no shared
    appearance space between one model's output and another's."""
    t = RefByteTracker()
    for _ in range(4):
        before = ids_left_to_right(t.update(
            [det(100, w=100.0, h=200.0, cls=0), det(125, w=100.0, h=200.0, cls=1)],
            [RED, BLUE]))
    after = ids_left_to_right(t.update(
        [det(100, w=100.0, h=200.0, cls=0), det(125, w=100.0, h=200.0, cls=1)],
        [BLUE, RED]))
    assert after == before


def test_empty_embeddings_fall_back_to_geometry():
    """An empty entry means 'no appearance for this detection' — the intended
    way to skip the ReID model on the cheap crops."""
    t = RefByteTracker()
    for _ in range(4):
        before = ids_left_to_right(t.update(
            [det(100, w=100.0, h=200.0), det(125, w=100.0, h=200.0)], [[], []]))
    after = ids_left_to_right(t.update(
        [det(100, w=100.0, h=200.0), det(125, w=100.0, h=200.0)], [[], []]))
    assert after == before


def test_embedding_count_must_match_detections():
    t = RefByteTracker()
    with pytest.raises(Exception):
        t.update([det(100), det(300)], [RED])


def test_embedding_width_must_not_change_mid_stream():
    t = RefByteTracker()
    t.update([det(100)], [RED])
    with pytest.raises(Exception):
        t.update([det(100)], [[1.0, 0.0]])


def test_ema_alpha_must_be_a_fraction():
    with pytest.raises(Exception):
        RefByteTracker(RefByteTrackConfig(ema_alpha=1.5))


def test_ema_alpha_rises_toward_one_for_a_weak_detection():
    """A half-occluded crop must barely move the appearance template: the
    effective smoothing factor goes to 1 (= no update) as the score falls to the
    high/low split."""
    t = RefByteTracker()
    assert t._ema_alpha(1.0) == pytest.approx(0.95)
    assert t._ema_alpha(0.5) == pytest.approx(1.0)
    assert t._ema_alpha(0.75) == pytest.approx(0.975)
    assert t._ema_alpha(0.2) == pytest.approx(1.0)  # clamped, not extrapolated


# =========================================================================== #
# Camera motion and the BoostTrack++ switches                                 #
# =========================================================================== #
def test_camera_motion_preserves_id_across_a_pan():
    """A camera pan moves every box at once. Motion alone reads that as every
    target teleporting and drops the tracks; warping the tracklets by the
    measured camera transform first keeps them."""
    def run(with_gmc):
        t = RefByteTracker()
        for _ in range(5):
            ids = [x.track_id for x in t.update([det(100, w=100.0, h=200.0)])]
        shift = 260.0  # well beyond the box width
        if with_gmc:
            t.apply_camera_motion([1, 0, shift, 0, 1, 0])
        after = [x.track_id for x in t.update([det(100 + shift, w=100.0, h=200.0)])]
        return ids, after

    before_no, after_no = run(False)
    before_yes, after_yes = run(True)
    assert before_no == before_yes and len(before_no) == 1
    assert after_no != before_no, "without GMC the pan should break the track"
    assert after_yes == before_yes, "with GMC the id should survive the pan"


def test_boost_flags_are_off_by_default():
    """The whole ablation depends on a default-constructed config meaning
    'none of this'."""
    b = RefByteTrackConfig().boost
    assert not b.rich_similarity and not b.soft_biou and not b.boost_detections


def test_detection_boost_lets_a_weak_detection_refind_a_lost_track():
    """ByteTrack's low-score pool can only rescue tracks that are still TRACKED;
    a lost track can only be re-found from the high pool. So a weak detection
    sitting exactly where a lost track was predicted is wasted — unless the
    tracklet vouches for it and the boost promotes it."""
    def run(boost):
        cfg = RefByteTrackConfig()
        cfg.boost.boost_detections = boost
        t = RefByteTracker(cfg)
        for _ in range(6):
            ids = [x.track_id for x in t.update([det(100, w=100.0, h=200.0)])]
        t.update([])  # miss a frame -> the track is lost
        after = [x.track_id for x in
                 t.update([det(100, w=100.0, h=200.0, score=0.35)])]
        return ids, after

    before_off, after_off = run(False)
    before_on, after_on = run(True)
    assert len(before_off) == 1 and before_off == before_on
    assert after_off != before_off, "unboosted, the weak detection cannot re-find it"
    assert after_on == before_on, "boosted, the same detection re-finds the track"


def test_duo_boost_starts_a_track_from_an_unexplained_detection():
    """A weak detection that no tracklet can explain is more likely a new object
    than noise, so DUO promotes it past the track-creation threshold."""
    def run(boost):
        cfg = RefByteTrackConfig()
        cfg.boost.boost_detections = boost
        t = RefByteTracker(cfg)
        for _ in range(5):
            t.update([det(100, w=100.0, h=200.0)])
        for _ in range(3):
            tracks = t.update([det(100, w=100.0, h=200.0),
                               det(900, w=100.0, h=200.0, score=0.45)])
        return len(tracks)

    assert run(False) == 1, "unboosted, the weak far detection should not track"
    assert run(True) == 2, "DUO should promote it into a track"


def test_boost_switches_do_not_break_plain_tracking():
    """Turning everything on must still track a simple moving object."""
    cfg = RefByteTrackConfig()
    cfg.boost.rich_similarity = True
    cfg.boost.soft_biou = True
    cfg.boost.boost_detections = True
    t = RefByteTracker(cfg)
    ids = set()
    for f in range(12):
        for trk in t.update([det(100 + 8 * f, w=100.0, h=200.0)]):
            ids.add(trk.track_id)
    assert len(ids) == 1, f"a single object produced {len(ids)} ids: {sorted(ids)}"


def test_reset_restarts_ids_at_one():
    t = RefByteTracker()
    for _ in range(3):
        t.update([det(100), det(400)])
    assert {x.track_id for x in t.update([det(100), det(400)])} == {1, 2}
    t.reset()
    after = {x.track_id for x in t.update([det(100), det(400)])}
    assert after == {1, 2}, "after reset the id counter must start over"
    assert t.frame_id == 1


def test_zero_size_box_does_not_poison_the_assignment():
    """A detector can emit a degenerate box (a box entirely above the frame is
    clamped to y1 == y2 == 0). The aspect term would then be infinite and the
    whole cost row NaN, so the side is clamped at the source."""
    t = RefByteTracker()
    for _ in range(4):
        tracks = t.update([det(100, w=100.0, h=200.0),
                           {"x1": 50.0, "y1": 0.0, "x2": 50.0, "y2": 0.0,
                            "score": 0.9, "class_id": 0}])
        for x in tracks:
            assert math.isfinite(x.x1) and math.isfinite(x.y2)
    assert any(x.track_id == 1 for x in tracks), "the real object must survive"


# =========================================================================== #
# ReID crop preprocessing                                                     #
# =========================================================================== #
def test_reid_preprocess_shape_and_range():
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, (97, 43, 3), dtype=np.uint8)  # odd size on purpose
    out = ref_reid_preprocess(crop, 0, 0, 43, 97)
    assert out.shape == (1, 3, 256, 128) and out.dtype == np.float32
    assert np.isfinite(out).all()
    # ImageNet z-score of a [0,1] input lands in roughly [-2.2, 2.7].
    assert -2.2 <= out.min() and out.max() <= 2.7


def test_reid_preprocess_swaps_to_rgb():
    """A pure-blue crop must land in the BLUE channel of the output, i.e. index
    2 after the BGR->RGB swap. Catches the swap being dropped, which nothing
    about the array's shape or range would reveal."""
    blue = np.zeros((40, 20, 3), np.uint8)
    blue[..., 0] = 255  # BGR: the blue channel
    out = ref_reid_preprocess(blue, 0, 0, 20, 40)[0]
    assert out[2].mean() > out[0].mean() and out[2].mean() > out[1].mean()


def test_reid_preprocess_squashes_rather_than_pads():
    """A very wide crop must fill the whole canvas, not sit in a letterbox: an
    aspect-preserving resize would leave constant bars, and constant bars have
    zero variance."""
    rng = np.random.default_rng(1)
    wide = rng.integers(0, 256, (20, 400, 3), dtype=np.uint8)
    out = ref_reid_preprocess(wide, 0, 0, 400, 20)[0, 0]
    assert out.std(axis=1).min() > 0.05, "rows look like padding bars"
    assert out.std(axis=0).min() > 0.05, "columns look like padding bars"


def test_reid_preprocess_clips_a_box_hanging_off_the_frame():
    """Detector boxes routinely run past the edge; the crop must clip instead of
    reading out of bounds."""
    rng = np.random.default_rng(2)
    frame = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    out = ref_reid_preprocess(frame, -30.0, -20.0, 60.0, 90.0)
    assert out.shape == (1, 3, 256, 128) and np.isfinite(out).all()
    with pytest.raises(Exception):  # entirely outside -> empty after clipping
        ref_reid_preprocess(frame, 200.0, 200.0, 300.0, 300.0)


def test_reid_preprocess_matches_opencv_if_available():
    """cv2's statement of the same thing, when cv2 happens to be installed.

    Worth pinning because every convention here fails SILENTLY: BGR left
    unswapped, a letterbox instead of a squash, or the pixel-center offset
    dropped all yield a well-formed array and a plausible-looking embedding that
    simply matches the wrong people.
    """
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(3)
    crop = rng.integers(0, 256, (97, 43, 3), dtype=np.uint8)
    resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    want = ((rgb - np.array(IMAGENET_MEAN, np.float32)) /
            np.array(IMAGENET_STD, np.float32)).transpose(2, 0, 1)[None]
    got = ref_reid_preprocess(crop, 0, 0, 43, 97)
    assert np.abs(got - want).max() < 2e-2


def test_reid_preprocess_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "reid_preprocess"):
        pytest.skip("compiled rcdl module without reid_preprocess bindings")
    rng = np.random.default_rng(4)
    crop = rng.integers(0, 256, (97, 43, 3), dtype=np.uint8)
    got = np.asarray(cxx.reid_preprocess(crop), np.float32).reshape(1, 3, 256, 128)
    assert np.abs(got - ref_reid_preprocess(crop, 0, 0, 43, 97)).max() < 1e-3


# =========================================================================== #
# Cross-checks against the compiled tracker                                   #
# =========================================================================== #
def _scene_straight_line():
    return [[det(100 + 12 * f, y=50)] for f in range(20)]


def _scene_gap_and_return():
    frames = [[det(200, y=100)] for _ in range(5)]
    frames += [[] for _ in range(7)]
    frames += [[det(200, y=100)] for _ in range(3)]
    return frames


def _scene_crossing():
    return [[det(100 + 20 * f, y=80), det(500 - 20 * f, y=80)] for f in range(21)]


def _scene_weak_rescue():
    frames = [[det(100 + 10 * f, y=60)] for f in range(5)]
    frames += [[det(100 + 10 * f, y=60, score=0.35)] for f in range(5, 11)]
    return frames


@pytest.mark.parametrize("scene,kw", [
    (_scene_straight_line, {}),
    (_scene_gap_and_return, {"track_buffer": 5}),
    (_scene_crossing, {}),
    (_scene_weak_rescue, {}),
])
def test_tracker_matches_cxx(cxx, scene, kw):
    """Same frames, same ids. Both sides implement the same algorithm with the
    same tie-breaks, so the id streams must agree exactly, not merely in shape."""
    if cxx is None:
        pytest.skip("compiled rcdl module without ByteTracker bindings")
    frames = scene()
    assert _cxx_run(cxx, frames, **kw) == _ref_run(frames, **kw)
