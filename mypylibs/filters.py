
import numpy as np
from scipy import ndimage

LAPLACIAN = np.array([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]], dtype=float)

FEATURES = {
    'luma': ('Grayscale (Rec. 601 luma)', lambda rgb: rgb @ [.299, .587, .114]),
    'min': ('Grayscale (min of R, G, B)', lambda rgb: rgb.min(axis=2)),
}


def gaussian_kernel(sigma):
    radius = int(np.ceil(3 * sigma))
    axis = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def disk_kernel(radius):
    axis = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(axis, axis)
    return (xx**2 + yy**2 <= radius**2).astype(float)


def otsu_threshold(img):
    hist, edges = np.histogram(img, bins=256, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(hist)
    mass = np.cumsum(hist * centers)
    with np.errstate(divide='ignore', invalid='ignore'):
        between = (mass[-1] * weight / weight[-1] - mass)**2 / (weight * (weight[-1] - weight))
    return centers[np.nanargmax(between)]


def dilate(mask, radius):
    return ndimage.convolve(mask.astype(float), disk_kernel(radius), mode='nearest') > .5


def erode(mask, radius):
    kernel = disk_kernel(radius)
    return ndimage.convolve(mask.astype(float), kernel, mode='nearest') > kernel.sum() - .5


def keep_largest(mask, count):
    labels, found = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = ndimage.sum_labels(mask, labels, np.arange(1, found + 1))
    return np.isin(labels, np.argsort(sizes)[::-1][:count] + 1)


def extract_boundary(img, feature='luma', sigma=2., threshold=None, invert=False,
                     opening=2, closing=2, components=1, fill_holes=True):
    """Return the 1px boundary mask and the (title, image) steps that produced it."""
    rgb = np.asarray(img.convert('RGB'), dtype=float) / 255
    steps = [('Input image', rgb)]

    title, to_gray = FEATURES[feature]
    gray = to_gray(rgb)
    steps.append((title, gray))

    kernel = gaussian_kernel(sigma)
    blurred = ndimage.convolve(gray, kernel, mode='nearest')
    steps.append((f'Gaussian blur (σ={sigma}, {len(kernel)}×{len(kernel)})', blurred))

    steps.append(('Laplacian of blurred image (comparison)',
                  ndimage.convolve(blurred, LAPLACIAN, mode='nearest')))

    level = otsu_threshold(blurred) if threshold is None else threshold
    mask = blurred < level if invert else blurred > level
    source = 'Otsu' if threshold is None else 'fixed'
    steps.append((f'Threshold ({source}, {"<" if invert else ">"} {level:.2f})', mask))

    if opening:
        mask = dilate(erode(mask, opening), opening)
    if closing:
        mask = erode(dilate(mask, closing), closing)
    steps.append((f'Opening r={opening}, closing r={closing}', mask))

    mask = keep_largest(mask, components)
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    holes = 'holes filled' if fill_holes else 'holes kept'
    kept = f'Largest {components} components' if components > 1 else 'Largest component'
    steps.append((f'{kept}, {holes}', mask))

    response = ndimage.convolve(mask.astype(float), LAPLACIAN, mode='constant')
    steps.append(('Laplacian response', response))

    boundary = mask & (response > 0)
    steps.append((f'Boundary ({boundary.sum()} px)', boundary))
    return boundary, steps
