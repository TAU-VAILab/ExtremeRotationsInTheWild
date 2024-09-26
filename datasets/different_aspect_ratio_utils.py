##taken from dust3r
import numpy as np
import cv2
import PIL.Image
try:
    
    lanczos = PIL.Image.Resampling.LANCZOS
except AttributeError:
    lanczos = PIL.Image.LANCZOS
'''This code is taken from Dust3R - 
dust3r/datasets/utils/cropping.py, dust3r/datasets/base/base_stereo_view_dataset.py.
https://github.com/naver/dust3r.git
Reference:
[1] Shuzhe Wang, Vincent Leroy, Yohann Cabon, , Boris Chidlovskii, Jerome Revaud
    DUSt3R: Geometric 3D Vision Made Easy. arXiv:2312.14132
'''
class BatchRandomAspectRatio:
    def __init__(self):
        self.resolutions = [(256, 192), (256, 168), (256, 144), (256, 128)]
        self.chosen_resolution = None
        self.portrait = None
        
    def __call__(self):
        if self.chosen_resolution is None:
            self.chosen_resolution = self.resolutions[np.random.choice(range(len(self.resolutions)))]
            self.portrait = np.random.choice([True, False])


    def reset(self):
        self.chosen_resolution = None
        
class ImageList:
    """ Convenience class to aply the same operation to a whole set of images.
    """

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch('resize', *args, **kwargs))

    def crop(self, *args, **kwargs):
        return ImageList(self._dispatch('crop', *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]

def rescale_image_depthmap(image, camera_intrinsics, output_resolution):
    """ Jointly rescale a (image, depthmap) 
        so that (out_width, out_height) >= output_res
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    
    scale_final = max(output_resolution / image.size) + 1e-8
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # first rescale the image so that it contains the crop
    image = image.resize(output_resolution, resample=lanczos)

    # no offset here; simple rescaling
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)

    return image.to_pil(), camera_intrinsics


def camera_matrix_of_crop(input_camera_matrix, input_resolution, output_resolution, scaling=1, offset_factor=0.5, offset=None):
    # Margins to offset the origin
    out_camera_matrix = input_camera_matrix.copy()
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins
    # Generate new camera parameters
    out_camera_matrix[:2, :] *= scaling
    out_camera_matrix[:2, 2] -= offset
    

    return out_camera_matrix

def bbox_from_intrinsics_in_out(input_camera_matrix, output_camera_matrix, output_resolution):
    out_width, out_height = output_resolution
    l, t = np.int32(np.round(input_camera_matrix[:2, 2] - output_camera_matrix[:2, 2]))
    crop_bbox = (l, t, l+out_width, t+out_height)
    return crop_bbox

def crop_image_depthmap(image, camera_intrinsics, crop_bbox):
    """
    Return a crop of the input view.
    """
    image = ImageList(image)
    l, t, r, b = crop_bbox

    image = image.crop((l, t, r, b))

    camera_intrinsics = camera_intrinsics.copy()
    camera_intrinsics[0, 2] -= l
    camera_intrinsics[1, 2] -= t

    return image.to_pil(), camera_intrinsics
    
def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_cropped_image_w_ar(image,fov,resolution,portrait=False):
    W, H ,C = image.shape  # new size

    assert resolution[0] >= resolution[1]
    if portrait:
        # image is portrait mode
        resolution = resolution[::-1]
    target_resolution = np.array(resolution)
    fl = 0.5*W/np.tan(fov/2)
    intrinsics = np.array([[fl, 0, 0.5*W], [0, fl, 0.5*H], [0, 0, 1]])
    image, intrinsics = rescale_image_depthmap(image, intrinsics, target_resolution)
    intrinsics2 = camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, intrinsics2 = crop_image_depthmap(image, intrinsics, crop_bbox)
    pad_image = np.zeros((H,W,C),dtype = np.uint8)
    pad_image[round(H/2-intrinsics2[1,2]):round(H/2+intrinsics2[1,2]),round(W/2-intrinsics2[0,2]):round(W/2+intrinsics2[0,2]),:]=np.array(image)
    return pad_image



