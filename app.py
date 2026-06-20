"""Streamlit app: U-Net road segmentation + A* route planning.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference_engine import predict_mask
from pathfinding import astar

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    _HAS_SIC = True
except ImportError:
    _HAS_SIC = False

# ── constants ────────────────────────────────────────────────────────────────
MASK_SIZE = 512  # must match U-Net input resolution

PIN_RADIUS = 8
PIN_A_COLOR = (50, 150, 255, 255)   # blue
PIN_B_COLOR = (255, 165, 0, 255)    # orange
ROAD_TINT   = (0,   200,  80,  65)  # semi-transparent green overlay
OBS_TINT    = (180,   0,   0,  75)  # semi-transparent red overlay
PATH_COLOR  = (255, 215,   0, 220)  # yellow path
PATH_WIDTH  = 4

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Route Planner | U-Net",
    page_icon="🗺️",
    layout="wide",
)

# ── session state defaults ────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "orig_img":    None,   # PIL RGB resized to MASK_SIZE x MASK_SIZE
    "mask":        None,   # np.ndarray (H,W) uint8
    "point_a":     None,   # (x, y) in mask-pixel space
    "point_b":     None,   # (x, y) in mask-pixel space
    "path":        None,   # list[(row, col)] or None
    "click_mode":  "A",    # "A" or "B"
    "_last_click": None,   # last coords dict processed — guards against rerun replay
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── helpers ───────────────────────────────────────────────────────────────────

def _build_display_image() -> Image.Image:
    """Composite: original + segmentation overlay + path + pins."""
    base = st.session_state.orig_img.copy().convert("RGBA")
    mask = st.session_state.mask

    # Segmentation tint
    tint = np.zeros((MASK_SIZE, MASK_SIZE, 4), dtype=np.uint8)
    tint[mask == 1] = ROAD_TINT
    tint[mask == 0] = OBS_TINT
    base = Image.alpha_composite(base, Image.fromarray(tint, "RGBA"))

    # Path overlay
    if st.session_state.path:
        path_layer = Image.new("RGBA", (MASK_SIZE, MASK_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(path_layer)
        pts = [(c, r) for r, c in st.session_state.path]  # (col, row) → (x, y)
        if len(pts) > 1:
            draw.line(pts, fill=PATH_COLOR, width=PATH_WIDTH)
        # Draw small dots at endpoints of the path to cap the line cleanly
        for px, py in (pts[0], pts[-1]):
            r = PATH_WIDTH
            draw.ellipse([px - r, py - r, px + r, py + r], fill=PATH_COLOR)
        base = Image.alpha_composite(base, path_layer)

    # Pins
    draw = ImageDraw.Draw(base)
    for point, color, label in [
        (st.session_state.point_a, PIN_A_COLOR, "A"),
        (st.session_state.point_b, PIN_B_COLOR, "B"),
    ]:
        if point:
            x, y = point
            r = PIN_RADIUS
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="white", width=2)
            # Label beside pin
            draw.text((x + r + 3, y - r), label, fill="white")

    return base.convert("RGB")


def _reset_route() -> None:
    st.session_state.path = None


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🗺️ Route Planner")
    st.caption("U-Net segmentation + A* pathfinding")
    st.divider()

    # Step 1: upload & segment
    st.subheader("1  Upload & Segment")
    uploaded = st.file_uploader(
        "Satellite / drone image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded:
        if st.button("Run Segmentation", type="primary", use_container_width=True):
            raw = Image.open(uploaded).convert("RGB")
            with st.spinner("Running U-Net inference…"):
                mask = predict_mask(raw)

            # Apply Morphological Closing to stitch gaps up to ~15 pixels
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

            st.session_state.orig_img    = raw.resize((MASK_SIZE, MASK_SIZE), Image.BILINEAR)
            st.session_state.mask        = mask
            st.session_state.point_a     = None
            st.session_state.point_b     = None
            st.session_state.path        = None
            st.session_state.click_mode  = "A"
            st.session_state._last_click = None
            st.success("Segmentation complete!")

    st.divider()

    # Step 2: pin placement
    st.subheader("2  Place Pins")
    if st.session_state.mask is not None:
        mode_label = st.radio(
            "Next click sets:",
            ["A — Start", "B — End"],
            index=0 if st.session_state.click_mode == "A" else 1,
            horizontal=True,
        )
        st.session_state.click_mode = "A" if mode_label.startswith("A") else "B"

        col_a, col_b = st.columns(2)
        with col_a:
            if st.session_state.point_a:
                st.success(f"A  ({st.session_state.point_a[0]}, {st.session_state.point_a[1]})")
            else:
                st.info("A  —  not set")
        with col_b:
            if st.session_state.point_b:
                st.success(f"B  ({st.session_state.point_b[0]}, {st.session_state.point_b[1]})")
            else:
                st.info("B  —  not set")

        if st.button("Clear Pins", use_container_width=True):
            st.session_state.point_a     = None
            st.session_state.point_b     = None
            st.session_state.path        = None
            st.session_state.click_mode  = "A"
            st.session_state._last_click = None
            st.rerun()
    else:
        st.info("Segment an image first.")

    st.divider()

    # Step 3: pathfinding
    st.subheader("3  Find Route")
    route_ready = (
        st.session_state.mask is not None
        and st.session_state.point_a is not None
        and st.session_state.point_b is not None
    )
    if st.button("Find Route", disabled=not route_ready, type="primary", use_container_width=True):
        ax, ay = st.session_state.point_a
        bx, by = st.session_state.point_b
        with st.spinner("Running A* pathfinding…"):
            path = astar(
                st.session_state.mask,
                start=(ay, ax),   # convert (x,y) → (row,col)
                end=(by, bx),
            )
        st.session_state.path = path
        if path is None:
            st.error("No traversable path found between these points.")
        else:
            st.success(f"Route found — {len(path)} steps.")


# ── main panel ────────────────────────────────────────────────────────────────
if not _HAS_SIC:
    st.error(
        "`streamlit-image-coordinates` is not installed. "
        "Run `pip install streamlit-image-coordinates` and restart."
    )
    st.stop()

if st.session_state.mask is None:
    st.info("Upload a satellite image and click **Run Segmentation** in the sidebar to begin.")
    st.stop()

st.subheader("Interactive Map  —  click to set pins")
st.caption(
    f"Currently placing: **{'Start (A)' if st.session_state.click_mode == 'A' else 'End (B)'}**  "
    "· Change mode in the sidebar."
)

display = _build_display_image()

# Coordinate-capture widget — returns {x, y} on click, else None
coords = streamlit_image_coordinates(display, key="map_click")

if coords is not None and coords != st.session_state._last_click:
    # Guard: streamlit-image-coordinates replays the last click on every rerun.
    # Only process when coords is genuinely new.
    st.session_state._last_click = coords
    cx = max(0, min(MASK_SIZE - 1, int(coords["x"])))
    cy = max(0, min(MASK_SIZE - 1, int(coords["y"])))
    if st.session_state.click_mode == "A":
        st.session_state.point_a   = (cx, cy)
        st.session_state.click_mode = "B"
    else:
        st.session_state.point_b   = (cx, cy)
        st.session_state.click_mode = "A"
    _reset_route()
    st.rerun()

# Metrics row
if st.session_state.path is not None:
    m1, m2, m3 = st.columns(3)
    m1.metric("Path length (px)", len(st.session_state.path))
    # Approximate metres — DeepGlobe tiles are ~2250m wide at 2448px → ~0.92 m/px
    # At 512px display we rescale: 2250 / 512 ≈ 4.4 m/px
    m2.metric("Est. distance", f"{len(st.session_state.path) * 4.4:.0f} m")
    m3.metric("Mask resolution", f"{MASK_SIZE}×{MASK_SIZE}")
