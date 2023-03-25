from numpy import log, log2, ceil, floor, zeros, sqrt, array
from cv2 import resize, GaussianBlur, INTER_LINEAR, INTER_NEAREST, subtract

# def get_octaves_n(img_shape, discard_n=DISCARDED_OCTAVES_N):
#     '''compute the number of octaves. Top 3 octaves will be descarded by default due to small size'''
#     return int(ceil(log2(min(img_shape[0], img_shape[1]))) - discard_n)

def get_octaves_n(image_shape, discard):
    '''Compute number of octaves in image pyramid as function (OpenCV default)'''
    return int(round(log(min(image_shape)) / log(2) - discard))

def get_img_base(image, sigma, assumed_blur):
    '''Generate base image from input image by upsampling by 2 in both directions and blurring'''
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

def get_gaussian_kernels(num_intervals, sigma):
    '''return a list of gaussian kernels (octave 0)'''
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = zeros(num_images_per_octave)
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels

# def generateGaussianKernels(sigma, num_intervals):
#     """Generate list of gaussian kernels at which to blur the input image. Default values of sigma, intervals, and octaves follow section 3 of Lowe's paper."""
#     num_images_per_octave = num_intervals + 3
#     k = 2 ** (1. / num_intervals)
#     gaussian_kernels = zeros(num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
#     gaussian_kernels[0] = sigma

#     for image_index in range(1, num_images_per_octave):
#         sigma_previous = (k ** (image_index - 1)) * sigma
#         sigma_total = k * sigma_previous
#         gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
#     return gaussian_kernels

# def get_LoG(base, kernels, octaves_n=OCTAVES_N):
#     '''return all octaves given base iamge and kernels'''
#     logs = []
#     img = base
#     for i in range(0, octaves_n):
#         octave = []
#         octave.append(img) # base has been blurred
#         for kernel in kernels[1:]:
#             img = GaussianBlur(img, (0,0), sigmaX=kernel, sigmaY=kernel) 
#             octave.append(img)
#         logs.append(octave)
#         assert(len(octave) >= 3) # TODO: clean assert
#         octave_base = octave[-3]
#         img = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)
#     assert(len(logs) == octaves_n) # TODO: clean assert
#     return array(logs, dtype=object)

# def get_DoG(logs):
#     '''return DoGs image pyramid'''
#     dogs = []
#     for octave in logs:
#         scale_n = len(octave)
#         octave_dog = []
#         for i in range(1, scale_n):
#             octave_dog.append(subtract(octave[i], octave[i-1]))
#         assert(len(octave_dog) == scale_n-1) # TODO: clean assert
#         dogs.append(octave_dog)
#     return array(dogs, dtype=object)

# TODO: deprecated
def get_LoG(image, num_octaves, gaussian_kernels):
    '''return all octaves given base iamge and kernels'''
    gaussian_images = []

    for octave_index in range(num_octaves):
        gaussian_images_in_octave = []
        gaussian_images_in_octave.append(image)  # first image in octave already has the correct blur
        for gaussian_kernel in gaussian_kernels[1:]:
            image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(image)
        gaussian_images.append(gaussian_images_in_octave)
        octave_base = gaussian_images_in_octave[-3]
        image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)
    return array(gaussian_images, dtype=object)

# TODO: deprecated
def get_DoG(gaussian_images):
    '''return DoGs image pyramid''' 
    dog_images = []

    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(subtract(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
        dog_images.append(dog_images_in_octave)
    return array(dog_images, dtype=object)

