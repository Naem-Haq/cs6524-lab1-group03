
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mypylibs.figures import PERCENTS

COLS, ROWS = 5, 4
TILE_W = 320
TITLE_H = 26
MARGIN = 10
SUPERSAMPLE = 2
FPS = 24
NUM_FRAMES = 288
MIN_CIRCLE_RADIUS = 1.5

BACKGROUND = (255, 255, 255)
GUIDE = (150, 170, 190)
ARM = (200, 68, 43)
TRACE = (31, 78, 121)
TEXT = (40, 40, 40)


def _even(value):
    return int(value) // 2 * 2


def save_video(descriptors, path):
    """Render the 20-panel reconstruction mosaic with its epicycles spinning."""
    try:
        import imageio_ffmpeg
    except ImportError:
        print(f'  imageio-ffmpeg not installed, skipping {path}')
        return

    z = descriptors.z
    span_x, span_y = np.ptp(z.real), np.ptp(z.imag)
    scale = (TILE_W - 2 * MARGIN) / span_x
    # Clamp so that tall silhouettes do not produce an oversized video
    tile_h = int(min(span_y * scale, 1.2 * TILE_W)) + 2 * MARGIN + TITLE_H
    scale = min(scale, (tile_h - TITLE_H - 2 * MARGIN) / span_y)

    size = (_even(COLS * TILE_W), _even(ROWS * tile_h))
    canvas_size = (size[0] * SUPERSAMPLE, size[1] * SUPERSAMPLE)
    font = ImageFont.load_default(size=15 * SUPERSAMPLE)

    background = Image.new('RGB', canvas_size, BACKGROUND)
    draw = ImageDraw.Draw(background)
    center = complex(z.real.min() + span_x / 2, z.imag.min() + span_y / 2)
    panels = []
    for index, percent in enumerate(PERCENTS):
        K = descriptors.num_harmonics(percent)
        left, top = (index % COLS) * TILE_W, (index // COLS) * tile_h
        draw.text(((left + TILE_W / 2) * SUPERSAMPLE, (top + 4) * SUPERSAMPLE),
                  f'{percent}% (K = {K})', fill=TEXT, font=font, anchor='ma')
        origin = complex(left + TILE_W / 2, top + TITLE_H + (tile_h - TITLE_H) / 2)
        trace = _to_canvas(descriptors.reconstruct(K), center, scale, origin)
        panels.append((K, origin, trace))

    radii = np.abs(descriptors.F) / descriptors.N * scale * SUPERSAMPLE

    writer = imageio_ffmpeg.write_frames(path, size, fps=FPS, codec='libx264',
                                         quality=7, macro_block_size=2)
    writer.send(None)
    for frame in range(NUM_FRAMES):
        n = descriptors.N * (frame + 1) / NUM_FRAMES
        arm = descriptors.arm(n)
        image = background.copy()
        draw = ImageDraw.Draw(image)
        for K, origin, trace in panels:
            joints = np.concatenate((arm[:1], arm[:K]))
            _draw_panel(draw, K, trace, n, _to_canvas(joints, center, scale, origin), radii)
        writer.send(np.asarray(image.resize(size, Image.Resampling.BOX)).tobytes())
    writer.close()


def _to_canvas(points, center, scale, origin):
    return ((points - center) * scale + origin) * SUPERSAMPLE


def _draw_panel(draw, K, trace, n, joints, radii):
    # joints[i + 1] is the tip of harmonic i, whose circle is centred on joints[i]
    for i in range(1, K):
        radius = radii[i]
        if radius < MIN_CIRCLE_RADIUS * SUPERSAMPLE:
            continue
        cx, cy = joints[i].real, joints[i].imag
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=GUIDE, width=1)

    drawn = trace[:int(n) + 1]
    if len(drawn) > 1:
        draw.line(np.column_stack((drawn.real, drawn.imag)).ravel().tolist(), fill=TRACE,
                  width=2 * SUPERSAMPLE, joint='curve')
    draw.line(np.column_stack((joints.real, joints.imag)).ravel().tolist(), fill=ARM,
              width=SUPERSAMPLE)
