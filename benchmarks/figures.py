"""Drawing helpers for the benchmark's check figures.

A check figure exists to answer a question the timing table cannot: *did the
model find the right thing?* A row that says `22 ms` and `18 vehicles` is still
consistent with eighteen boxes in the sky. So every figure carries a caption
saying what it is evidence OF, and the drawing is deliberately literal — boxes
where boxes were reported, masks where masks were reported, nothing smoothed or
cherry-picked.

These live next to `bench.py` and are produced by the same run that produces the
numbers, so a figure cannot drift away from the row above it.

Needs OpenCV, like the examples do; without it `bench.py --figures` reports the
figures as skipped and still writes the table.
"""

import os

import numpy as np

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX, without importing cv2 at module scope
MAX_SIDE = 900   # figures are read in a README gallery at ~250 px wide
MIN_WIDTH = 520  # ...and clicked through at full size, where captions must be legible


def _cv2():
    import cv2
    return cv2


def put(img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    """Outlined text. A caption over a photograph is illegible in one colour —
    the black outline is what makes it readable over both sky and asphalt.

    The outline is drawn by repeating the SAME stroke thickness at one-pixel
    offsets rather than by a single fatter stroke, because in OpenCV 5 the glyph
    advance depends on thickness: the identical string measures 456 px at
    thickness 1 and 496 px at thickness 2+. A fatter halo therefore ends further
    right than the fill that is supposed to cover it, and the uncovered tail
    shows up as a ghost of the last few characters."""
    cv2 = _cv2()
    x, y = int(org[0]), int(org[1])
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                    thick, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def fit_line(img, text, base, thick=1, margin=20):
    """Make one line fit the width: shrink the font, then truncate if it still
    does not fit. Returns (text, scale).

    A caption that runs off the edge loses exactly the part that says what
    happened, and on a narrow figure — a cropped person, a strip of face crops —
    shrinking alone is not enough."""
    cv2 = _cv2()
    avail = img.shape[1] - margin

    def width(t, s):
        return cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, thick)[0][0]

    scale = base
    while scale > 0.32 and width(text, scale) > avail:
        scale -= 0.02
    if width(text, scale) <= avail:
        return text, scale
    while len(text) > 4 and width(text + "...", scale) > avail:
        text = text[:-1]
    return text + "...", scale


def caption(img, text, sub=None):
    """Bottom-left caption saying what the figure demonstrates, on its own strip.

    The strip is not decoration: these captions sit over photographs that are
    bright in some places and dark in others, and outlined text alone is
    readable over one and not the other."""
    return panel(img, [(text, (255, 255, 255))] + ([(sub, (190, 220, 255))] if sub else []),
                 bold_first=True)


def class_color(i):
    """Stable per-class colour, spread over the hue circle."""
    cv2 = _cv2()
    hsv = np.uint8([[[(int(i) * 67) % 180, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return int(b), int(g), int(r)


def box(img, x1, y1, x2, y2, label=None, color=(80, 220, 100), thick=2):
    cv2 = _cv2()
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
    if label:
        put(img, label, (int(x1) + 2, max(14, int(y1) - 6)), 0.55, color, 1)


def blend_mask(img, mask, color, alpha=0.45):
    """Tint the pixels a boolean mask marks, in place."""
    m = np.asarray(mask).astype(bool)
    if m.shape[:2] != img.shape[:2]:
        return img
    img[m] = (img[m] * (1.0 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)
    return img


def side_by_side(left, right, gap=8, labels=None):
    """Two images at the same height, with an optional label on each."""
    cv2 = _cv2()
    h = max(left.shape[0], right.shape[0])
    def fit(a):
        if a.shape[0] == h:
            return a
        s = h / a.shape[0]
        return cv2.resize(a, (int(round(a.shape[1] * s)), h))
    l, r = fit(left), fit(right)
    out = np.full((h, l.shape[1] + gap + r.shape[1], 3), 24, np.uint8)
    out[:, :l.shape[1]] = l
    out[:, l.shape[1] + gap:] = r
    if labels:
        put(out, labels[0], (10, 24), 0.6, (255, 255, 255), 2)
        put(out, labels[1], (l.shape[1] + gap + 10, 24), 0.6, (255, 255, 255), 2)
    return out


def panel(img, lines, alpha=0.82, bold_first=False):
    """Strip along the bottom carrying a few lines of text. Used where the
    picture alone cannot show the result — a similarity number, a class score —
    and a bare photograph would look like proof of nothing.

    Nearly opaque on purpose: at 0.6 the strip disappears over a bright subject
    and the text goes with it."""
    cv2 = _cv2()
    h, w = img.shape[:2]
    line_h = 24
    ph = 14 + line_h * len(lines)
    y0 = h - ph
    ov = img.copy()
    cv2.rectangle(ov, (0, y0), (w, h), (16, 16, 16), -1)
    cv2.addWeighted(ov, alpha, img, 1.0 - alpha, 0, img)
    for i, (text, color) in enumerate(lines):
        thick = 2 if (bold_first and i == 0) else 1
        text, scale = fit_line(img, text, 0.58 if thick == 2 else 0.52, thick)
        put(img, text, (10, y0 + 20 + line_h * i), scale, color, thick)
    return img


def bars(img, rows, title=None):
    """A small horizontal bar chart in the bottom strip — for the tasks whose
    output is a number rather than a place (classification scores, timings)."""
    cv2 = _cv2()
    h, w = img.shape[:2]
    ph = 20 + 26 * len(rows) + (22 if title else 0)
    y0 = h - ph
    ov = img.copy()
    cv2.rectangle(ov, (0, y0), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.82, img, 0.18, 0, img)
    y = y0 + 18
    if title:
        put(img, title, (10, y), 0.56, (255, 255, 255), 2)
        y += 22
    vmax = max((v for _, v, _ in rows), default=1.0) or 1.0
    x0, bw_max = 130, w - 130 - 90
    for label, v, color in rows:
        put(img, label[:18], (10, y + 12), 0.5, (220, 220, 220), 1)
        bw = max(2, int(bw_max * (v / vmax)))
        cv2.rectangle(img, (x0, y + 2), (x0 + bw, y + 16), color, -1)
        put(img, f"{v:.3f}" if v < 10 else f"{v:.0f}", (x0 + bw + 8, y + 14), 0.5,
            (255, 255, 255), 1)
        y += 26
    return img


def save(fig, out_dir, name):
    """Write one figure, capped at MAX_SIDE and floored at MIN_WIDTH.

    The floor matters as much as the cap: a figure built from a narrow crop
    comes out ~240 px wide, and three lines of caption on a 240 px strip are
    unreadable at any font size. Upscaling costs nothing here — these are
    evidence, not photographs."""
    cv2 = _cv2()
    os.makedirs(out_dir, exist_ok=True)
    if fig.shape[1] < MIN_WIDTH:
        s = MIN_WIDTH / fig.shape[1]
        fig = cv2.resize(fig, (MIN_WIDTH, int(round(fig.shape[0] * s))),
                         interpolation=cv2.INTER_CUBIC)
    longest = max(fig.shape[:2])
    if longest > MAX_SIDE:
        s = MAX_SIDE / longest
        fig = cv2.resize(fig, (int(round(fig.shape[1] * s)), int(round(fig.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
    path = os.path.join(out_dir, f"{name}.jpg")
    cv2.imwrite(path, fig, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path
