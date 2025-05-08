import torch
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

colab = False

model_spec = "facebook/sam2-hiera-large"
device = "cuda" if colab else "cpu"

mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(model_spec, device=device)
predictor = SAM2ImagePredictor.from_pretrained(model_spec, device=device)

def generate_masks(image):
    return mask_generator.generate(image)

def predict(image, pt_coords, pt_labels, multimask_output=True):
    predictor.set_image(image)
    return predictor.predict(point_coords=pt_coords, point_labels=pt_labels, multimask_output=multimask_output)

def generate_mask_image(masks):
    mask_img = _show_mask(masks)
    return _mask_image_post_processing(mask_img)

def _mask_image_post_processing(mask_img):
    n, m, k = mask_img.shape
    slicing = np.ones((mask_img.shape), dtype=bool)
    slicing[:, :, 3] = False
    return np.floor(mask_img[slicing].reshape((n, m, k-1)) * 255)

def _show_anns(anns, borders=True):
    np.random.seed(3)
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax = plt.gca()
    # ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    # ax.imshow(img)
    return img

def _show_mask(mask, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    return mask_image

