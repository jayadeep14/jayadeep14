#!/usr/bin/env python3

from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import argparse


# ============================================================
# IMAGE / MASK UTILITIES
# ============================================================

def largest_component(binary):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, 8
    )

    if count <= 1:
        return binary

    largest = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    return np.where(
        labels == largest,
        255,
        0
    ).astype(np.uint8)


def build_subject_mask(image):
    """
    Create a foreground mask using OpenCV GrabCut.
    """

    rgb = np.array(
        image.convert("RGB")
    )

    h, w = rgb.shape[:2]

    # Smaller image = faster GrabCut
    seg_w = min(520, w)

    seg_h = max(
        1,
        round(h * seg_w / w)
    )

    small = cv2.resize(
        rgb,
        (seg_w, seg_h),
        interpolation=cv2.INTER_AREA
    )

    bgr = cv2.cvtColor(
        small,
        cv2.COLOR_RGB2BGR
    )

    # Start as probable background
    mask = np.full(
        (seg_h, seg_w),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    # Force borders to background
    border = max(
        5,
        int(seg_w * 0.018)
    )

    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    cx = seg_w // 2

    # --------------------------------------------------------
    # HEAD / FACE PROBABLE FOREGROUND
    # --------------------------------------------------------

    cv2.ellipse(
        mask,
        (
            cx - 15,
            int(seg_h * 0.46)
        ),
        (
            int(seg_w * 0.22),
            int(seg_h * 0.19)
        ),
        0,
        0,
        360,
        cv2.GC_PR_FGD,
        -1
    )

    # --------------------------------------------------------
    # TORSO PROBABLE FOREGROUND
    # --------------------------------------------------------

    torso = np.array(
        [
            [
                int(seg_w * 0.10),
                int(seg_h * 0.65)
            ],
            [
                int(seg_w * 0.90),
                int(seg_h * 0.65)
            ],
            [
                int(seg_w * 0.98),
                int(seg_h * 0.98)
            ],
            [
                int(seg_w * 0.02),
                int(seg_h * 0.98)
            ]
        ],
        dtype=np.int32
    )

    cv2.fillConvexPoly(
        mask,
        torso,
        cv2.GC_PR_FGD
    )

    # --------------------------------------------------------
    # CERTAIN FACE REGION
    # --------------------------------------------------------

    cv2.ellipse(
        mask,
        (
            cx + 8,
            int(seg_h * 0.48)
        ),
        (
            max(1, int(seg_w * 0.06)),
            max(1, int(seg_h * 0.13))
        ),
        0,
        0,
        360,
        cv2.GC_FGD,
        -1
    )

    bg_model = np.zeros(
        (1, 65),
        np.float64
    )

    fg_model = np.zeros(
        (1, 65),
        np.float64
    )

    # --------------------------------------------------------
    # GRABCUT
    # --------------------------------------------------------

    try:
        cv2.grabCut(
            bgr,
            mask,
            None,
            bg_model,
            fg_model,
            7,
            cv2.GC_INIT_WITH_MASK
        )
    except cv2.error:
        # Fallback if GrabCut cannot initialize
        fallback = np.zeros(
            (seg_h, seg_w),
            dtype=np.uint8
        )

        cv2.ellipse(
            fallback,
            (
                cx,
                int(seg_h * 0.45)
            ),
            (
                int(seg_w * 0.25),
                int(seg_h * 0.25)
            ),
            0,
            0,
            360,
            255,
            -1
        )

        cv2.fillConvexPoly(
            fallback,
            torso,
            255
        )

        return Image.fromarray(
            fallback
        ).resize(
            (w, h),
            Image.Resampling.LANCZOS
        )

    # --------------------------------------------------------
    # BINARY MASK
    # --------------------------------------------------------

    binary = np.where(
        (
            (mask == cv2.GC_FGD) |
            (mask == cv2.GC_PR_FGD)
        ),
        255,
        0
    ).astype(np.uint8)

    # Keep largest connected subject
    binary = largest_component(
        binary
    )

    # --------------------------------------------------------
    # CLEAN MASK
    # --------------------------------------------------------

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        np.ones(
            (7, 7),
            np.uint8
        ),
        iterations=2
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        np.ones(
            (3, 3),
            np.uint8
        ),
        iterations=1
    )

    binary = cv2.GaussianBlur(
        binary,
        (5, 5),
        0
    )

    return Image.fromarray(
        binary
    ).resize(
        (w, h),
        Image.Resampling.LANCZOS
    )


def cover_crop_box(w, h, ratio):
    """
    Center-crop image to target aspect ratio.
    """

    src = w / h

    if src > ratio:
        # Image is wider
        cw = int(h * ratio)

        left = (
            w - cw
        ) // 2

        return (
            left,
            0,
            left + cw,
            h
        )

    # Image is taller
    ch = int(w / ratio)

    top = (
        h - ch
    ) // 2

    return (
        0,
        top,
        w,
        top + ch
    )


# ============================================================
# SVG GENERATOR
# ============================================================

def generate(image_path, output_path):

    # --------------------------------------------------------
    # CANVAS
    # --------------------------------------------------------

    D_W = 1400
    D_H = 600

    # --------------------------------------------------------
    # PORTRAIT PANEL
    # --------------------------------------------------------

    P_W = 460
    P_H = 480

    P_X = 850
    P_Y = 60

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGBA")

    mask = build_subject_mask(
        image
    )

    # --------------------------------------------------------
    # CROP
    # --------------------------------------------------------

    crop = cover_crop_box(
        image.width,
        image.height,
        P_W / P_H
    )

    gray = image.convert(
        "L"
    ).crop(crop)

    subject = mask.crop(
        crop
    )

    # --------------------------------------------------------
    # DOT GRID
    # --------------------------------------------------------

    SPACING = 4

    cols = P_W // SPACING
    rows = P_H // SPACING

    gray_small = ImageEnhance.Contrast(
        gray.resize(
            (cols, rows),
            Image.Resampling.LANCZOS
        )
    ).enhance(1.35)

    mask_small = subject.resize(
        (cols, rows),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # ANIMATION
    # --------------------------------------------------------

    DURATION = 10.0

    circles = []

    total = max(
        1,
        rows * cols - 1
    )

    # --------------------------------------------------------
    # GENERATE PORTRAIT PARTICLES
    # --------------------------------------------------------

    for y in range(rows):

        for x in range(cols):

            mask_value = mask_small.getpixel(
                (x, y)
            )

            # Background
            if mask_value < 125:
                continue

            lum = (
                gray_small.getpixel(
                    (x, y)
                ) / 255.0
            )

            # ------------------------------------------------
            # DOT SIZE
            # ------------------------------------------------

            radius = (
                0.40 +
                (lum ** 0.90) * 1.15
            )

            # ------------------------------------------------
            # DOT OPACITY
            # ------------------------------------------------

            opacity = (
                0.42 +
                lum * 0.58
            )

            # ------------------------------------------------
            # DOT POSITION
            # ------------------------------------------------

            cx = (
                P_X +
                x * SPACING +
                SPACING / 2
            )

            cy = (
                P_Y +
                y * SPACING +
                SPACING / 2
            )

            # ------------------------------------------------
            # REVEAL TIMING
            # ------------------------------------------------

            raster = (
                y * cols + x
            ) / total

            appear = (
                1.0 +
                raster * 5.4
            )

            appear2 = min(
                appear + 0.10,
                7.8
            )

            t1 = appear / DURATION
            t2 = appear2 / DURATION

            # Clamp values for valid SVG keyTimes
            t1 = max(
                0.0,
                min(0.82, t1)
            )

            t2 = max(
                t1,
                min(0.82, t2)
            )

            key_times = (
                "0;"
                f"{t1:.6f};"
                f"{t2:.6f};"
                "0.82;"
                "1"
            )

            larger_radius = (
                radius + 0.35
            )

            circle = f"""
        <circle
            cx="{cx:.2f}"
            cy="{cy:.2f}"
            r="{radius:.2f}"
            fill="#78ff9c"
            fill-opacity="{opacity:.3f}"
            opacity="0">

            <animate
                attributeName="opacity"
                values="0;0;1;1;0"
                keyTimes="{key_times}"
                dur="{DURATION}s"
                repeatCount="indefinite"/>

            <animate
                attributeName="r"
                values="{radius:.2f};{radius:.2f};{larger_radius:.2f};{radius:.2f};{radius:.2f}"
                keyTimes="{key_times}"
                dur="{DURATION}s"
                repeatCount="indefinite"/>

        </circle>
"""

            circles.append(
                circle
            )

    circles_svg = "".join(
        circles
    )

    # ========================================================
    # SVG TEMPLATE
    #
    # IMPORTANT:
    # This is NOT an f-string.
    #
    # Therefore CSS { } are completely safe.
    # ========================================================

    svg_template = r'''<?xml version="1.0" encoding="UTF-8"?>

<svg
    xmlns="http://www.w3.org/2000/svg"
    width="__D_W__"
    height="__D_H__"
    viewBox="0 0 __D_W__ __D_H__"
    version="1.1">

<title>Jayadeep T P — Terminal Profile</title>

<desc>
Animated terminal developer profile with portrait particle animation.
</desc>


<defs>

    <!-- ====================================================
         MAIN GLOW
         ==================================================== -->

    <filter
        id="glow"
        x="-30%"
        y="-30%"
        width="160%"
        height="160%">

        <feGaussianBlur
            stdDeviation="2.5"
            result="blur"/>

        <feMerge>
            <feMergeNode in="blur"/>
            <feMergeNode in="SourceGraphic"/>
        </feMerge>

    </filter>


    <!-- ====================================================
         STRONG GLOW
         ==================================================== -->

    <filter
        id="strongGlow"
        x="-50%"
        y="-50%"
        width="200%"
        height="200%">

        <feGaussianBlur
            stdDeviation="5"
            result="blur"/>

        <feMerge>
            <feMergeNode in="blur"/>
            <feMergeNode in="blur"/>
            <feMergeNode in="SourceGraphic"/>
        </feMerge>

    </filter>


    <!-- ====================================================
         SCANNER GRADIENT
         ==================================================== -->

    <linearGradient
        id="scanGradient"
        x1="0%"
        y1="0%"
        x2="100%"
        y2="0%">

        <stop
            offset="0%"
            stop-color="#78ff9c"
            stop-opacity="0"/>

        <stop
            offset="45%"
            stop-color="#78ff9c"
            stop-opacity="0.2"/>

        <stop
            offset="50%"
            stop-color="#78ff9c"
            stop-opacity="1"/>

        <stop
            offset="55%"
            stop-color="#78ff9c"
            stop-opacity="0.2"/>

        <stop
            offset="100%"
            stop-color="#78ff9c"
            stop-opacity="0"/>

    </linearGradient>


    <!-- ====================================================
         PORTRAIT GRADIENT
         ==================================================== -->

    <linearGradient
        id="portraitGlow"
        x1="0%"
        y1="0%"
        x2="0%"
        y2="100%">

        <stop
            offset="0%"
            stop-color="#78ff9c"/>

        <stop
            offset="50%"
            stop-color="#52d77d"/>

        <stop
            offset="100%"
            stop-color="#1d6b38"/>

    </linearGradient>


    <!-- ====================================================
         CSS
         ==================================================== -->

    <style type="text/css"><![CDATA[

        .term-text {
            font-family:
                "Courier New",
                Courier,
                monospace;

            fill: #78ff9c;
            filter: url(#glow);
        }

        .title {
            font-family:
                "Courier New",
                Courier,
                monospace;

            font-size: 52px;
            font-weight: bold;
            letter-spacing: 2px;
            fill: #78ff9c;
        }

        .subtitle {
            font-family:
                "Courier New",
                Courier,
                monospace;

            font-size: 18px;
            font-weight: bold;
            letter-spacing: 1px;
            fill: #52c478;
        }

        .tag {
            font-family:
                "Courier New",
                Courier,
                monospace;

            font-size: 15px;
            font-weight: bold;
            fill: #78ff9c;
            text-anchor: middle;
        }

        .comment {
            font-family:
                "Courier New",
                Courier,
                monospace;

            font-size: 16px;
            fill: #48a867;
            opacity: 0.75;
        }

        .meta {
            font-family:
                "Courier New",
                Courier,
                monospace;

            font-size: 15px;
            fill: #48a867;
            opacity: 0.8;
        }

        .status {
            font-family:
                "Courier New",
                Courier,
                monospace;

            font-size: 16px;
            font-weight: bold;
            fill: #78ff9c;
        }

        .terminal-line {
            stroke: #48a867;
            fill: none;
        }

    ]]></style>

</defs>


<!-- ========================================================
     BACKGROUND
     ======================================================== -->

<rect
    x="0"
    y="0"
    width="__D_W__"
    height="__D_H__"
    fill="#030a05"/>


<!-- Background grid -->

<g
    opacity="0.08"
    stroke="#48a867"
    stroke-width="1">

    <path d="M 30 100 H 1370"/>
    <path d="M 30 200 H 1370"/>
    <path d="M 30 300 H 1370"/>
    <path d="M 30 400 H 1370"/>
    <path d="M 30 500 H 1370"/>

</g>


<!-- ========================================================
     OUTER FRAME
     ======================================================== -->

<rect
    x="15"
    y="15"
    width="__OUTER_W__"
    height="__OUTER_H__"
    rx="15"
    fill="#000000"
    stroke="#1d4227"
    stroke-width="3"/>

<rect
    x="22"
    y="22"
    width="__INNER_W__"
    height="__INNER_H__"
    rx="10"
    fill="none"
    stroke="#0b1c10"
    stroke-width="1"/>


<!-- ========================================================
     TERMINAL HEADER
     ======================================================== -->

<g transform="translate(65, 52)">

    <circle
        cx="5"
        cy="0"
        r="3"
        fill="#78ff9c">

        <animate
            attributeName="opacity"
            values="1;0.25;1"
            dur="1.5s"
            repeatCount="indefinite"/>

    </circle>

    <text
        x="18"
        y="5"
        class="meta">

        TERMINAL_PROFILE.exe

    </text>

    <text
        x="285"
        y="5"
        class="meta">

        SYSTEM ONLINE

    </text>

</g>


<!-- ========================================================
     LEFT PROFILE
     ======================================================== -->

<g transform="translate(65, 105)">

    <!-- HELLO -->

    <text
        class="subtitle"
        x="0"
        y="0">

        &gt; HELLO, WORLD

        <animate
            attributeName="opacity"
            values="0;0;1;1"
            keyTimes="0;0.05;0.15;1"
            dur="10s"
            repeatCount="indefinite"/>

    </text>


    <!-- FIRST NAME -->

    <text
        class="title"
        x="0"
        y="65">

        Jayadeep T P

        <animate
            attributeName="opacity"
            values="0;0;1;1"
            keyTimes="0;0.15;0.28;1"
            dur="10s"
            repeatCount="indefinite"/>

        <animate
            attributeName="transform"
            values="translate(-20 0);translate(-20 0);translate(0 0);translate(0 0)"
            keyTimes="0;0.15;0.28;1"
            dur="10s"
            repeatCount="indefinite"/>

    </text>


    <!-- LAST NAME -->

    <text
        class="title"
        x="0"
        y="125">

        

        <tspan
            fill="#3cfc73"
            stroke="#3cfc73"
            stroke-width="2">

            _

            <animate
                attributeName="opacity"
                values="1;0;1"
                dur="0.8s"
                repeatCount="indefinite"/>

        </tspan>

        <animate
            attributeName="opacity"
            values="0;0;1;1"
            keyTimes="0;0.20;0.34;1"
            dur="10s"
            repeatCount="indefinite"/>

    </text>


    <!-- PROFESSION -->

    <text
        class="subtitle"
        x="0"
        y="185">

        Linux | python vibe coder | AI Builder 

        <animate
            attributeName="opacity"
            values="0;0;1;1"
            keyTimes="0;0.30;0.42;1"
            dur="10s"
            repeatCount="indefinite"/>

    </text>


    <!-- ====================================================
         SKILL TAGS
         ==================================================== -->

    <g transform="translate(0, 225)">

        <!-- Java AI -->

        <g>

            <rect
                x="0"
                y="0"
                width="105"
                height="42"
                rx="10"
                fill="#06130a"
                stroke="#48a867"
                stroke-width="1.5"
                stroke-dasharray="3,3">

                <animate
                    attributeName="opacity"
                    values="0;0;1;1"
                    keyTimes="0;0.40;0.48;1"
                    dur="10s"
                    repeatCount="indefinite"/>

            </rect>

            <text
                class="tag"
                x="52.5"
                y="26">

                Java AI

                <animate
                    attributeName="opacity"
                    values="0;0;1;1"
                    keyTimes="0;0.40;0.48;1"
                    dur="10s"
                    repeatCount="indefinite"/>

            </text>

        </g>


        <!-- Robotics -->

        <g>

            <rect
                x="120"
                y="0"
                width="135"
                height="42"
                rx="10"
                fill="#06130a"
                stroke="#48a867"
                stroke-width="1.5"
                stroke-dasharray="3,3">

                <animate
                    attributeName="opacity"
                    values="0;0;1;1"
                    keyTimes="0;0.44;0.52;1"
                    dur="10s"
                    repeatCount="indefinite"/>

            </rect>

            <text
                class="tag"
                x="187.5"
                y="26">

                Robotics

                <animate
                    attributeName="opacity"
                    values="0;0;1;1"
                    keyTimes="0;0.44;0.52;1"
                    dur="10s"
                    repeatCount="indefinite"/>

            </text>

        </g>


        <!-- PyQt6 / C++ -->

        <g>

            <rect
                x="270"
                y="0"
                width="115"
                height="42"
                rx="10"
                fill="#06130a"
                stroke="#48a867"
                stroke-width="1.5"
                stroke-dasharray="3,3">

                <animate
                    attributeName="opacity"
                    values="0;0;1;1"
                    keyTimes="0;0.48;0.56;1"
                    dur="10s"
                    repeatCount="indefinite"/>

            </rect>

            <text
                class="tag"
                x="327.5"
                y="26">

                PyQt6 / C++

                <animate
                    attributeName="opacity"
                    values="0;0;1;1"
                    keyTimes="0;0.48;0.56;1"
                    dur="10s"
                    repeatCount="indefinite"/>

            </text>

        </g>


        <!-- LLM -->

        <g>

            <rect
                x="400"
                y="0"
                width="75"
                height="42"
                rx="10"
                fill="#06130a"
                stroke="#48a867"
                stroke-width="1.5"
                stroke-dasharray="3,3">

                <animate
                    attributeName="opacity"
                    values="0;0;1;1"
                    keyTimes="0;0.52;0.60;1"
                    dur="10s"
                    repeatCount="indefinite"/>

            </rect>

            <text
                class="tag"
                x="437.5"
                y="26">

                LLM

                <animate
                    attributeName="opacity"
                    values="0;0;1;1"
                    keyTimes="0;0.52;0.60;1"
                    dur="10s"
                    repeatCount="indefinite"/>

            </text>

        </g>


        <!-- AI Products -->

        <g>

            <rect
                x="490"
                y="0"
                width="145"
                height="42"
                rx="10"
                fill="#06130a"
                stroke="#48a867"
                stroke-width="1.5"
                stroke-dasharray="3,3">

                <animate
                    attributeName="opacity"
                    values="0;0;1;1"
                    keyTimes="0;0.56;0.64;1"
                    dur="10s"
                    repeatCount="indefinite"/>

            </rect>

            <text
                class="tag"
                x="562.5"
                y="26">

                AI Products

                <animate
                    attributeName="opacity"
                    values="0;0;1;1"
                    keyTimes="0;0.56;0.64;1"
                    dur="10s"
                    repeatCount="indefinite"/>

            </text>

        </g>

    </g>


    <!-- DESCRIPTION -->

    <text
        class="comment"
        x="0"
        y="340">

        | visual. + secure APIs, SaaS products, custom tooling

        <animate
            attributeName="opacity"
            values="0;0;0.75;0.75"
            keyTimes="0;0.60;0.72;1"
            dur="10s"
            repeatCount="indefinite"/>

    </text>


    <!-- COMMAND -->

    <text
        class="meta"
        x="0"
        y="385">

        &gt; initializing profile...

        <animate
            attributeName="opacity"
            values="1;0.2;1"
            dur="1.8s"
            repeatCount="indefinite"/>

    </text>

</g>


<!-- ========================================================
     PORTRAIT FRAME
     ======================================================== -->

<rect
    x="__FRAME_X__"
    y="__FRAME_Y__"
    width="__FRAME_W__"
    height="__FRAME_H__"
    rx="8"
    fill="#020704"
    stroke="#132e1b"
    stroke-width="2"/>


<!-- Inner portrait border -->

<rect
    x="__INNER_FRAME_X__"
    y="__INNER_FRAME_Y__"
    width="__INNER_FRAME_W__"
    height="__INNER_FRAME_H__"
    rx="5"
    fill="none"
    stroke="#0d2917"
    stroke-width="1"/>


<!-- ========================================================
     SCANNER CORNERS
     ======================================================== -->

<path
    d="
        M __CORNER1_X__ __CORNER1_Y1__
        L __CORNER1_X__ __CORNER1_Y2__
        L __CORNER1_X2__ __CORNER1_Y2__
    "
    class="terminal-line"
    stroke-width="2"/>


<path
    d="
        M __CORNER2_X__ __CORNER2_Y1__
        L __CORNER2_X__ __CORNER2_Y2__
        L __CORNER2_X2__ __CORNER2_Y2__
    "
    class="terminal-line"
    stroke-width="2"/>


<!-- ========================================================
     VERTICAL SCANNER
     ======================================================== -->

<rect
    x="__P_X__"
    y="__P_Y__"
    width="8"
    height="__P_H__"
    fill="url(#scanGradient)"
    opacity="0">

    <animate
        attributeName="x"
        values="__P_X__;__SCAN_END_X__;__P_X__"
        dur="4s"
        begin="0s"
        repeatCount="indefinite"/>

    <animate
        attributeName="opacity"
        values="0;0.9;0"
        dur="4s"
        repeatCount="indefinite"/>

</rect>


<!-- ========================================================
     HORIZONTAL SCANNER
     ======================================================== -->

<line
    x1="__P_X__"
    y1="__P_Y__"
    x2="__P_END_X__"
    y2="__P_Y__"
    stroke="#78ff9c"
    stroke-width="1.5"
    opacity="0">

    <animate
        attributeName="y1"
        values="__P_Y__;__P_END_Y__;__P_Y__"
        dur="3.5s"
        repeatCount="indefinite"/>

    <animate
        attributeName="y2"
        values="__P_Y__;__P_END_Y__;__P_Y__"
        dur="3.5s"
        repeatCount="indefinite"/>

    <animate
        attributeName="opacity"
        values="0;0.8;0"
        dur="3.5s"
        repeatCount="indefinite"/>

</line>


<!-- ========================================================
     SCANNING STATUS
     ======================================================== -->

<text
    class="status"
    x="__STATUS_X__"
    y="__STATUS_Y__"
    text-anchor="end">

    SCANNING

    <animate
        attributeName="opacity"
        values="1;0.3;1"
        dur="0.8s"
        repeatCount="indefinite"/>

</text>


<!-- ========================================================
     COMPLETE INDICATOR
     ======================================================== -->

<g>

    <circle
        cx="__COMPLETE_DOT_X__"
        cy="__COMPLETE_DOT_Y__"
        r="3"
        fill="#78ff9c">

        <animate
            attributeName="opacity"
            values="0;0;1;1"
            keyTimes="0;0.60;0.75;1"
            dur="10s"
            repeatCount="indefinite"/>

    </circle>

    <text
        class="status"
        x="__COMPLETE_TEXT_X__"
        y="__COMPLETE_TEXT_Y__"
        text-anchor="end">

        COMPLETE

        <animate
            attributeName="opacity"
            values="0;0;1;1"
            keyTimes="0;0.60;0.75;1"
            dur="10s"
            repeatCount="indefinite"/>

    </text>

</g>


<!-- ========================================================
     PORTRAIT DOTS
     ======================================================== -->

<g
    id="portrait"
    filter="url(#glow)">

__CIRCLES__

</g>


<!-- ========================================================
     PORTRAIT GLOW PULSE
     ======================================================== -->

<rect
    x="__P_X__"
    y="__P_Y__"
    width="__P_W__"
    height="__P_H__"
    fill="none"
    stroke="#78ff9c"
    stroke-width="1"
    opacity="0.05">

    <animate
        attributeName="opacity"
        values="0.02;0.10;0.02"
        dur="2.5s"
        repeatCount="indefinite"/>

</rect>


<!-- ========================================================
     FOOTER
     ======================================================== -->

<text
    class="meta"
    x="850"
    y="572">

    &gt; portrait.svg loaded // terminal profile

    <animate
        attributeName="opacity"
        values="0.4;1;0.4"
        dur="2s"
        repeatCount="indefinite"/>

</text>


<!-- ========================================================
     SYSTEM INDICATORS
     ======================================================== -->

<g transform="translate(65, 555)">

    <circle
        cx="0"
        cy="0"
        r="3"
        fill="#78ff9c">

        <animate
            attributeName="opacity"
            values="1;0.2;1"
            dur="1s"
            repeatCount="indefinite"/>

    </circle>

    <text
        class="meta"
        x="12"
        y="5">

        SECURE CONNECTION

    </text>


    <circle
        cx="205"
        cy="0"
        r="3"
        fill="#52c478">

        <animate
            attributeName="opacity"
            values="1;0.2;1"
            dur="1.4s"
            repeatCount="indefinite"/>

    </circle>

    <text
        class="meta"
        x="217"
        y="5">

        AI ENGINE READY

    </text>


    <circle
        cx="405"
        cy="0"
        r="3"
        fill="#3cfc73">

        <animate
            attributeName="opacity"
            values="1;0.2;1"
            dur="1.8s"
            repeatCount="indefinite"/>

    </circle>

    <text
        class="meta"
        x="417"
        y="5">

        JAVA RUNTIME

    </text>

</g>

</svg>
'''

    # ========================================================
    # REPLACE SAFE PLACEHOLDERS
    # ========================================================

    replacements = {
        "__D_W__": str(D_W),
        "__D_H__": str(D_H),

        "__OUTER_W__": str(D_W - 30),
        "__OUTER_H__": str(D_H - 30),

        "__INNER_W__": str(D_W - 44),
        "__INNER_H__": str(D_H - 44),

        "__FRAME_X__": str(P_X - 30),
        "__FRAME_Y__": str(P_Y - 2),
        "__FRAME_W__": str(P_W + 60),
        "__FRAME_H__": str(P_H + 4),

        "__INNER_FRAME_X__": str(P_X - 20),
        "__INNER_FRAME_Y__": str(P_Y + 8),
        "__INNER_FRAME_W__": str(P_W + 40),
        "__INNER_FRAME_H__": str(P_H - 16),

        "__CORNER1_X__": str(P_X - 35),
        "__CORNER1_Y1__": str(P_Y + 30),
        "__CORNER1_Y2__": str(P_Y - 7),
        "__CORNER1_X2__": str(P_X + 10),

        "__CORNER2_X__": str(P_X + P_W + 35),
        "__CORNER2_Y1__": str(P_Y + P_H - 30),
        "__CORNER2_Y2__": str(P_Y + P_H + 7),
        "__CORNER2_X2__": str(P_X + P_W - 10),

        "__P_X__": str(P_X),
        "__P_Y__": str(P_Y),
        "__P_W__": str(P_W),
        "__P_H__": str(P_H),

        "__P_END_X__": str(P_X + P_W),
        "__P_END_Y__": str(P_Y + P_H),

        "__SCAN_END_X__": str(P_X + P_W),

        "__STATUS_X__": str(P_X + P_W + 5),
        "__STATUS_Y__": str(P_Y + 32),

        "__COMPLETE_DOT_X__": str(P_X + P_W - 8),
        "__COMPLETE_DOT_Y__": str(P_Y + 22),

        "__COMPLETE_TEXT_X__": str(P_X + P_W - 18),
        "__COMPLETE_TEXT_Y__": str(P_Y + 27),

        "__CIRCLES__": circles_svg,
    }

    for key, value in replacements.items():
        svg_template = svg_template.replace(
            key,
            value
        )

    svg = svg_template

    # ========================================================
    # BASIC XML VALIDATION
    # ========================================================

    try:
        import xml.etree.ElementTree as ET

        ET.fromstring(svg)

    except Exception as exc:
        raise RuntimeError(
            f"Generated SVG is not valid XML: {exc}"
        ) from exc

    # ========================================================
    # WRITE SVG
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        svg,
        encoding="utf-8"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate an animated "
            "terminal-style developer "
            "profile SVG from a portrait."
        )
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Input portrait image"
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(
            "terminal_profile.svg"
        ),
        help="Output SVG file"
    )

    args = parser.parse_args()

    try:

        generate(
            args.image,
            args.output
        )

        print(
            f"Generated successfully: "
            f"{args.output}"
        )

    except Exception as exc:

        print(
            f"ERROR: {exc}"
        )

        raise


if __name__ == "__main__":
    main()

