import numpy as np
from functools import cmp_to_key
from PIL import Image
import glob

__all__ = ["VisualizationCF"]


class VisualizationCF:
    @staticmethod
    def convert_to_image(obs_np):
        assert type(obs_np) is np.ndarray
        intersection_num = obs_np.shape[1]
        new_shape = (
            obs_np.shape[0],
            intersection_num * obs_np.shape[2],
            intersection_num * obs_np.shape[3],
        )
        return np.reshape(obs_np, new_shape).T.astype(np.uint8)

    @staticmethod
    def save_gif(path, frame_size):
        def cmp_func(item1, item2):
            item1 = item1.replace(f"{path}/", "")
            item2 = item2.replace(f"{path}/", "")
            item1 = item1.replace(".png", "")
            item2 = item2.replace(".png", "")
            item1 = int(item1)
            item2 = int(item2)
            return item1 - item2

        images_path = glob.glob(f"{path}/*.png")
        file_path = f"{path}/movie.gif"
        imgs = (Image.open(f) for f in sorted(images_path, key=cmp_to_key(cmp_func)))
        img = next(imgs)  # extract first image from iterator
        img.save(
            fp=file_path,
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=100,
            loop=0,
        )
