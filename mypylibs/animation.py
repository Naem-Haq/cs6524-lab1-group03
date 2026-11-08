
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH = 800
TITLE_H = 40
MARGIN = 24
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
    """Render the full-harmonic reconstruction being drawn by its spinning epicycles."""
    try:
        import imageio_ffmpeg
    except ImportError:
        print(f'  imageio-ffmpeg not installed, skipping {path}')
        return

    z = descriptors.z
    K = descriptors.N
    span_x, span_y = np.ptp(z.real), np.ptp(z.imag)
    scale = (WIDTH - 2 * MARGIN) / span_x
    # Clamp so that tall silhouettes do not produce an oversized video
    height = int(min(span_y * scale, 1.2 * WIDTH)) + 2 * MARGIN + TITLE_H
    scale = min(scale, (height - TITLE_H - 2 * MARGIN) / span_y)

    size = (_even(WIDTH), _even(height))
    canvas_size = (size[0] * SUPERSAMPLE, size[1] * SUPERSAMPLE)
    font = ImageFont.load_default(size=20 * SUPERSAMPLE)

    background = Image.new('RGB', canvas_size, BACKGROUND)
    ImageDraw.Draw(background).text((WIDTH / 2 * SUPERSAMPLE, 10 * SUPERSAMPLE),
                                     f'100% (K = {K})', fill=TEXT, font=font, anchor='ma')
    center = complex(z.real.min() + span_x / 2, z.imag.min() + span_y / 2)
    origin = complex(WIDTH / 2, TITLE_H + (height - TITLE_H) / 2)
    trace = _to_canvas(descriptors.reconstruct(K), center, scale, origin)
    radii = np.abs(descriptors.F) / descriptors.N * scale * SUPERSAMPLE

    writer = imageio_ffmpeg.write_frames(path, size, fps=FPS, codec='libx264',
                                         quality=7, macro_block_size=2)
    writer.send(None)
    for frame in range(NUM_FRAMES):
        n = descriptors.N * (frame + 1) / NUM_FRAMES
        arm = descriptors.arm(n)
        joints = _to_canvas(np.concatenate((arm[:1], arm)), center, scale, origin)
        image = background.copy()
        _draw_frame(ImageDraw.Draw(image), trace, n, joints, radii)
        writer.send(np.asarray(image.resize(size, Image.Resampling.BOX)).tobytes())
    writer.close()


def _to_canvas(points, center, scale, origin):
    return ((points - center) * scale + origin) * SUPERSAMPLE


def _draw_frame(draw, trace, n, joints, radii):
    # joints[i + 1] is the tip of harmonic i, whose circle is centred on joints[i]
    for i in range(1, len(radii)):
        radius = radii[i]
        if radius < MIN_CIRCLE_RADIUS * SUPERSAMPLE:
            continue
        cx, cy = joints[i].real, joints[i].imag
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=GUIDE,
                     width=SUPERSAMPLE)

    drawn = trace[:int(n) + 1]
    if len(drawn) > 1:
        draw.line(np.column_stack((drawn.real, drawn.imag)).ravel().tolist(), fill=TRACE,
                  width=2 * SUPERSAMPLE, joint='curve')
    draw.line(np.column_stack((joints.real, joints.imag)).ravel().tolist(), fill=ARM,
              width=SUPERSAMPLE)
