from SIFT.Pyramid.buildScaleSpace import get_img_base, get_octaves_n, get_gaussian_kernels, get_LoG, get_DoG
from SIFT.Keypoint.extrema import findScaleSpaceExtrema
from SIFT.Keypoint.clean import deduplicate_keypoints, postsolve_keypoints
from SIFT.Descriptor.genDescriptor import generateDescriptors
from utils.stat_time import *
from utils.params import SIGMA_ZERO, SCALE_N, DISCARDED_OCTAVES_N

DEBUG_MODE = 1
# def get_keypoints_descriptors(img):
#     float_img = img.astype('float32')
#     # tic
#     timer = Timer('generating descriptors')

#     r, base = get_img_base(float_img, assumed_blur=1)
#     kernels = get_gaussian_kernels()
#     logs = get_LoG(base, kernels=kernels)
#     dogs = get_DoG(logs)

#     keypoints = peak_detection(logs, dogs)
#     keypoints = deduplicate_keypoints(keypoints)
#     keypoints = postsolve_keypoints(keypoints, r)
#     for i in keypoints:
#         print(i.pt, i.size, i.angle)

#     descriptors = get_descriptors(keypoints, logs)
#     # for i in descriptors:
#     #     print(i)
    
#     # tac
#     timer.tac()
#     print("{} keypoints, {} descriptors".format(len(keypoints), len(descriptors)))
#     return keypoints, descriptors

def get_keypoints_descriptors(image, sigma=SIGMA_ZERO, scale_n=SCALE_N, assumed_blur=1, image_border_width=5):
    """Compute SIFT keypoints and descriptors for an input image
    """
    # timer = Timer('generating descriptors')

    image = image.astype('float32')
    base_image = get_img_base(image, sigma, assumed_blur)
    num_octaves = get_octaves_n(base_image.shape, discard=DISCARDED_OCTAVES_N)
    gaussian_kernels = get_gaussian_kernels(scale_n-3, sigma)
    gaussian_images = get_LoG(base_image, num_octaves, gaussian_kernels)
    dog_images = get_DoG(gaussian_images)
    keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, scale_n-3, sigma, image_border_width)
    keypoints = deduplicate_keypoints(keypoints)
    keypoints = postsolve_keypoints(keypoints)
    # if DEBUG_MODE:
    #     for keypoint in keypoints:
    #         print(keypoint.pt)
    descriptors = generateDescriptors(keypoints, gaussian_images)
    if DEBUG_MODE:
        for descriptor in descriptors:
            print(descriptor)
    # timer.tac()    
    return keypoints, descriptors