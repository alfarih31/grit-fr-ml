import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 300
image_mean = np.array([110.763, 110.499, 106.5])  # RGB layout
image_std =  1.0 #np.array([78.77, 80.508, 86.448])
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(38, 8, SSDBoxSizes(25, 57), [2]),
    SSDSpec(19, 16, SSDBoxSizes(57, 111), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
    SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
]

priors = generate_ssd_priors(specs, image_size)
