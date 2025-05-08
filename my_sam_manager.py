import numpy as np
import os


class MySamManager:
    def __init__(self, image_obj, dir=".", storage=None):
        self.image = image_obj["image"]
        self.filename = image_obj["filename"]
        self.mask_image = np.zeros((self.image.shape[:-1]), dtype=bool)
        self.mask_images = list()
        self.storage = os.path.join(dir, "labels", self.filename) if storage is None else storage

    def store_mask(self, mask):
        self.mask_images.append(mask)  
    
    def update_mask(self, mask_bool):
        self.mask_image |= mask_bool

    def persist(self):
        with open(self.storage, "wb") as out_file:
            for mask in self.mask_image:
                np.save(out_file, mask)
    
    def exists(self, x, y):
        return self.mask_image[y, x]
    
