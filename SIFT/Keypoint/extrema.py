from utils.params import CONTRAST_THRESHOLD, RADIUS_FACTOR, PEAK_RATIO, BINS_N, SIGMA_ZERO, FLOAT_TOLERANCE, SCALE_N, SIGMA_ZERO, EIGENVALUE_RATIO
from numpy import floor, array, stack, dot, trace, float32, zeros, sqrt, arctan2, rad2deg, deg2rad, exp, logical_and, roll, where
from numpy.linalg import det, lstsq
from cv2 import KeyPoint
import logging

logger = logging.getLogger(__name__)

def get_gradient(pixel_array):
    '''approximate gradient at center pixel [1,1,1], where dx = [f(x+h)-f(x-h)] / 2h 
       x => second, y => first, z => third
    '''
    dx = (pixel_array[1,1,2]-pixel_array[1,1,0]) / 2.0
    dy = (pixel_array[1,2,1]-pixel_array[1,0,1]) / 2.0
    dz = (pixel_array[2,1,1]-pixel_array[0,1,1]) / 2.0
    return array([dx, dy, dz])

def get_hessian(pixel_array):
    '''approximate Hessian matrix at center pixel [1,1,1]'''
    dxx = pixel_array[1,1,2]+pixel_array[1,1,0]-2*pixel_array[1,1,1]
    dyy = pixel_array[1,2,1]+pixel_array[1,0,1]-2*pixel_array[1,1,1]
    dzz = pixel_array[2,1,1]+pixel_array[0,1,1]-2*pixel_array[1,1,1]
    dxy = (pixel_array[1,2,2]+pixel_array[1,0,0]-pixel_array[1,2,0]-pixel_array[1,0,2]) / 4.0
    dxz = (pixel_array[2,1,2]+pixel_array[0,1,0]-pixel_array[2,1,0]-pixel_array[0,1,2]) / 4.0
    dyz = (pixel_array[2,2,1]+pixel_array[0,0,1]-pixel_array[0,2,1]-pixel_array[2,0,1]) / 4.0
    return array(
        [[dxx,dxy,dxz],
         [dxy,dyy,dyz],
         [dxz,dyz,dzz]]
    )

def isPeak(region1, region2, region3, threshold):
    '''return true if the center pixel of numpy array region 2'''
    center_value = region2[1,1]
    if abs(center_value) > threshold:
        if center_value > 0:
            # return all(center_value >= region1) and \
            #        all(center_value >= region3) and \
            #        all(center_value >= region2[0,:]) and \
            #        all(center_value >= region2[2,:]) and \
            #        center_value >= region2[1,0] and \
            #        center_value >= region2[1,2] 
            return region1.any() <= center_value and \
                   region3.any() <= center_value and \
                   region2[0,:].any() <= center_value and \
                   region2[2,:].any() <= center_value and \
                   region1[1,0].any() <= center_value and \
                   region1[1,2].any() <= center_value
        elif center_value < 0:
            # return all(center_value <= region1) and \
            #        all(center_value <= region3) and \
            #        all(center_value <= region2[0,:]) and \
            #        all(center_value <= region2[2,:]) and \
            #        center_value <= region2[1,0] and \
            #        center_value <= region2[1,2]
            return region1.any() >= center_value and \
                   region3.any() >= center_value and \
                   region2[0,:].any() >= center_value and \
                   region2[2,:].any() >= center_value and \
                   region1[1,0].any() >= center_value and \
                   region1[1,2].any() >= center_value
    return False 

def keypoint_localization(x, y, img_idx, octave_idx, octave_dog, border_width=5, max_iter=5, scale_n=SCALE_N, sigma=SIGMA_ZERO, contrast_threshold=CONTRAST_THRESHOLD, eigenvalue_ratio=EIGENVALUE_RATIO):
    '''iteratively refine positions of extrema'''
    n = scale_n - 3
    out_of_img = False
    img_shape = octave_dog[0].shape
    for iter in range(max_iter):
        img1, img2, img3 = octave_dog[img_idx-1:img_idx+2]
        pixel_cube = stack(
            [img1[x-1:x+2,y-1:y+2],
            img2[x-1:x+2,y-1:y+2],
            img3[x-1:x+2,y-1:y+2]]
        ).astype('float32') / 255
        gradient = get_gradient(pixel_cube)
        hessian = get_hessian(pixel_cube)
        extrema_update = -lstsq(hessian,gradient,rcond=None)[0]
        # break if converges
        if abs(extrema_update[0])<0.5 and abs(extrema_update[1])<0.5 and abs(extrema_update[2])<0.5:
            break
        y += int(round(extrema_update[0]))
        x += int(round(extrema_update[1]))
        img_idx += int(round(extrema_update[2]))
        # check in image
        if x<border_width or x>=img_shape[0]-border_width or \
           y<border_width or y>=img_shape[1]-border_width or \
           img_idx<1 or img_idx>n:
            out_of_img = True
            break
    # check break
    if out_of_img:
        return None
    if iter >= max_iter - 1:
        return None
    f_extrema = pixel_cube[1,1,1] + dot(gradient, extrema_update) / 2.0        
    if abs(f_extrema) * n >= contrast_threshold:
        xy_hessian = hessian[:2,:2]
        xy_hessian_det = det(xy_hessian)
        xy_hessian_trace = trace(xy_hessian)
        if xy_hessian_det>0 and eigenvalue_ratio*(xy_hessian_trace**2)<((eigenvalue_ratio+1)**2)*xy_hessian_det:
            keypoint = KeyPoint()
            keypoint.pt = ((y+extrema_update[0])*(2**octave_idx), (x+extrema_update[1])*(2**octave_idx))
            keypoint.size = sigma * (2**((img_idx+extrema_update[2]) / float32(n))) * (2**(octave_idx+1))
            keypoint.response = abs(f_extrema)
            return keypoint, img_idx
    return None

def keypoint_orientation(keypoint, octave_idx, log, radius_factor=RADIUS_FACTOR, bins_n=BINS_N, peak_ratio=PEAK_RATIO, scale_factor=SIGMA_ZERO):
    '''compute orientations for keypoints'''
    kp_with_orientations = []
    img_shape = log.shape

    scale = scale_factor * keypoint.size / float32(2 ** (octave_idx+1))
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)

    raw_histogram = zeros(bins_n)
    smoothed_histogram = zeros(bins_n)

    for i in range(-radius, radius+1):
        y = int(round(keypoint.pt[1] / float32(2 ** octave_idx))) + i
        if y>0 and y<img_shape[0]-1:
            for j in range(-radius, radius+1):
                x = int(round(keypoint.pt[0] / float32(2 ** octave_idx))) + i
                if x>0 and x<img_shape[1]-1:
                    dx, dy = log[y,x+1] - log[y,x-1], log[y-1,x]-log[y+1,x]
                    magnitude = sqrt(dx*dx + dy*dy)
                    orientation = rad2deg(arctan2(dy,dx))
                    weight = exp(weight_factor*(i**2 + j**2))
                    idx = int(round(orientation * bins_n / 360.0))
                    raw_histogram[idx % bins_n] += weight * magnitude
    # smooth
    for i in range(bins_n):
        # [1, 4, 6, 4, 1]
        smoothed_histogram[i] = (6*raw_histogram[i] + 4*(raw_histogram[i-1]+raw_histogram[(i+1)%bins_n]) + raw_histogram[i-2] + raw_histogram[(i+2)%bins_n]) / 16.0
        orientation_max = max(smoothed_histogram)
        orientation_peaks = where(logical_and(smoothed_histogram > roll(smoothed_histogram, 1), smoothed_histogram > roll(smoothed_histogram, -1)))[0]
        for peak in orientation_peaks:
            value = smoothed_histogram[peak]
            if value >= peak_ratio * orientation_max:
                left, right = smoothed_histogram[(peak-1)%bins_n], smoothed_histogram[(peak+1)%bins_n]
                interpolation = (peak + 0.5*(left-right)/(left+right-2*value)) % bins_n
                kp_orientation = 360.0 - interpolation * 360.0 / bins_n
                if abs(kp_orientation - 360.0) < FLOAT_TOLERANCE:
                    kp_orientation = 0.0
                kp_with_orientations.append(KeyPoint(*keypoint.pt, keypoint.size, kp_orientation, keypoint.response, keypoint.octave))
    return kp_with_orientations

def peakDetection(logs, dogs, border_width=5, scale_n=SCALE_N, sigma=SIGMA_ZERO, contrast_threshold=CONTRAST_THRESHOLD):
    '''detecting the peak in 26 pixels - utils: keypoint_localization, keypoint_orientation'''
    n = scale_n - 3
    threshold = floor(0.5 * contrast_threshold / n * 255) # TODO: follow openCV implementation
    keypoints = []

    for octave_idx, octave in enumerate(dogs):
        print("octave " + str(octave_idx))
        for image_idx, (image1, image2, image3) in enumerate(zip(octave, octave[1:], octave[2:])):
            print("image" + str(image_idx))
            for x in range(border_width, image1.shape[0]-border_width):
                for y in range(border_width, image1.shape[1]-border_width):
                    if isPeak(image1[x-1:x+2,y-1:y+2] , image2[x-1:x+2,y-1:y+2], image3[x-1:x+2,y-1:y+2], threshold):
                        localization_result = keypoint_localization(x,y,image_idx+1,octave_idx,octave)
                        if localization_result is not None:
                            keypoint, localized_img_idx = localization_result
                            kp_with_orientations = keypoint_orientation(keypoint, octave_idx, logs[octave_idx][localized_img_idx])
                            for kp in kp_with_orientations:
                                keypoints.append(kp)
    return keypoints

def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
    """Find pixel positions of all scale-space extrema in the image pyramid
    """
    logger.debug('Finding scale-space extrema...')
    threshold = floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        print("octave " + str(octave_index))
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # (i, j) is the center of the 3x3 array
            print("image " + str(image_index))
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    # print((i,j))
                    if isPeak(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                        localization_result = localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints

# def isPixelAnExtremum(first_subimage, second_subimage, third_subimage, threshold):
#     """Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors, False otherwise
#     """
#     center_pixel_value = second_subimage[1, 1]
#     if abs(center_pixel_value) > threshold:
#         if center_pixel_value > 0:
#             return all(center_pixel_value >= first_subimage) and \
#                    all(center_pixel_value >= third_subimage) and \
#                    all(center_pixel_value >= second_subimage[0, :]) and \
#                    all(center_pixel_value >= second_subimage[2, :]) and \
#                    center_pixel_value >= second_subimage[1, 0] and \
#                    center_pixel_value >= second_subimage[1, 2]
#         elif center_pixel_value < 0:
#             return all(center_pixel_value <= first_subimage) and \
#                    all(center_pixel_value <= third_subimage) and \
#                    all(center_pixel_value <= second_subimage[0, :]) and \
#                    all(center_pixel_value <= second_subimage[2, :]) and \
#                    center_pixel_value <= second_subimage[1, 0] and \
#                    center_pixel_value <= second_subimage[1, 2]
#     return False

def localizeExtremumViaQuadraticFit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
    """Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors
    """
    logger.debug('Localizing scale-space extrema...')
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
        first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
        pixel_cube = stack([first_image[i-1:i+2, j-1:j+2],
                            second_image[i-1:i+2, j-1:j+2],
                            third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        gradient = computeGradientAtCenterPixel(pixel_cube)
        hessian = computeHessianAtCenterPixel(pixel_cube)
        extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        # make sure the new pixel_cube will lie entirely within the image
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        logger.debug('Updated extremum moved outside of image before reaching convergence. Skipping...')
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        logger.debug('Exceeded maximum number of attempts without reaching convergence for this extremum. Skipping...')
        return None
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * dot(gradient, extremum_update)
    if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = trace(xy_hessian)
        xy_hessian_det = det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            # Contrast check passed -- construct and return OpenCV KeyPoint object
            keypoint = KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, image_index
    return None

def computeGradientAtCenterPixel(pixel_array):
    """Approximate gradient at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    """
    # With step size h, the central difference formula of order O(h^2) for f'(x) is (f(x + h) - f(x - h)) / (2 * h)
    # Here h = 1, so the formula simplifies to f'(x) = (f(x + 1) - f(x - 1)) / 2
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    return array([dx, dy, ds])

def computeHessianAtCenterPixel(pixel_array):
    """Approximate Hessian at center pixel [1, 1, 1] of 3x3x3 array using central difference formula of order O(h^2), where h is the step size
    """
    # With step size h, the central difference formula of order O(h^2) for f''(x) is (f(x + h) - 2 * f(x) + f(x - h)) / (h ^ 2)
    # Here h = 1, so the formula simplifies to f''(x) = f(x + 1) - 2 * f(x) + f(x - 1)
    # With step size h, the central difference formula of order O(h^2) for (d^2) f(x, y) / (dx dy) = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ^ 2)
    # Here h = 1, so the formula simplifies to (d^2) f(x, y) / (dx dy) = (f(x + 1, y + 1) - f(x + 1, y - 1) - f(x - 1, y + 1) + f(x - 1, y - 1)) / 4
    # NOTE: x corresponds to second array axis, y corresponds to first array axis, and s (scale) corresponds to third array axis
    center_pixel_value = pixel_array[1, 1, 1]
    dxx = pixel_array[1, 1, 2] - 2 * center_pixel_value + pixel_array[1, 1, 0]
    dyy = pixel_array[1, 2, 1] - 2 * center_pixel_value + pixel_array[1, 0, 1]
    dss = pixel_array[2, 1, 1] - 2 * center_pixel_value + pixel_array[0, 1, 1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return array([[dxx, dxy, dxs], 
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])

#########################
# Keypoint orientations #
#########################

def computeKeypointsWithOrientations(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    """Compute orientations for each keypoint
    """
    logger.debug('Computing keypoint orientations...')
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    scale = scale_factor * keypoint.size / float32(2 ** (octave_index + 1))  # compare with keypoint.size computation in localizeExtremumViaQuadraticFit()
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = zeros(num_bins)
    smooth_histogram = zeros(num_bins)

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = sqrt(dx * dx + dy * dy)
                    gradient_orientation = rad2deg(arctan2(dy, dx))
                    weight = exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    orientation_max = max(smooth_histogram)
    orientation_peaks = where(logical_and(smooth_histogram > roll(smooth_histogram, 1), smooth_histogram > roll(smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            # Quadratic peak interpolation
            # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < FLOAT_TOLERANCE:
                orientation = 0
            new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations

