
import itertools
import multiprocessing
import os
import random
from typing import Generator

import numpy as np
import png  # type: ignore

import pyqoi


def test_png_progressive(route: str) -> tuple[str, bool | None]:
    try:
        p = png.Reader(filename=route)
        width, height, rgba_rows_flat_channels, info = p.asRGBA()
        image = np.array(list(itertools.chain.from_iterable(rgba_rows_flat_channels)), np.uint8)
        encoded = pyqoi.encode(image, width, height, colorspace=0)
        decoded = pyqoi.decode(encoded).data
        return (route, all((a == b for a, b in zip(decoded, image))))
    except Exception:
        return (route, None)

def manufactured_test():
    raw_image = [pyqoi.STARTING_VALUE] * 192
    raw_image[0:40:] = (((240 + i) % 256, (240 + i) % 256, (5 - i) % 256, 255) for i in range(40))
    raw_image[55:60:] = ((100, 100, 100, 100) for _ in range(5))
    raw_image[70:80:] = ((50, 50, 50, 50) for _ in range(10))
    raw_image[100:120:] = ((50 + i, 50 - i, 50, 50) for i in range(20))
    raw_image[120:140:] = ((100 + 1 * i, 100 - 2 * i, 50, 50) for i in range(20))
    raw_image[140:150:] = ((100, 100, 50 + (i % 2), 50) for i in range(10))
    raw_image[150:155:] = ((0, 0, 0, i) for i in range(5))
    raw_image[155:165:] = ((255, 0, 255, 0) if i % 2 else (0, 255, 0, 255) for i in range(10))
    raw_image[170:190:] = ((255 - 4 * i, 255 - 5 * i, 255, 255) for i in range(20))

    image = np.array([channel for pixel in raw_image for channel in pixel], dtype=np.uint8)

    encoded = pyqoi.encode(image, width=12, height=16, colorspace=0)
    decoded = pyqoi.decode(encoded).data
    assert all(a == b for a, b in zip(image, decoded)), "did not pass manufactured test"

def get_pngs(root: str) -> Generator[str, None, None]:
    for path, _, files in os.walk(root):
        for file in files:
            if file.endswith(".png"):
                yield f"{path}\\{file}"

def library_test():

    IMAGES_ROUTE = ".\\qoi_benchmark_suite\\images\\"

    pngs = list(get_pngs(IMAGES_ROUTE))
    random.shuffle(pngs)

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = pool.map(test_png_progressive, pngs[::])
    fl = list(filter(lambda x: x[1] is not True, results))
    print(f"Fail List: {fl}")
    if (len(fl) == 0):
        print("PyQOI passed all tests")
    else:
        print("PyQOI failed some tests")
        print("note: Htdamg_affinity.png fails in the standard library because the source file is malformed and cannot be loaded.")

def main():
    manufactured_test()
    library_test()

if __name__ == "__main__":
    main()
