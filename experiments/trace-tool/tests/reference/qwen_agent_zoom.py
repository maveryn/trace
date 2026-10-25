"""Reference copy of Qwen-Agent's ImageZoomInToolQwen3VL, used only by tests.

Source: https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/tools/image_zoom_in_qwen3vl.py
(Apache-2.0, Copyright 2023 The Qwen team, Alibaba Group). Methods copied
verbatim; call reduced to its crop/resize core (no file I/O or messages).
"""

from math import ceil, floor
import math

from PIL import Image


class ReferenceZoom:
    # Image resizing functions (copied from qwen-vl-utils)
    def round_by_factor(self, number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    def ceil_by_factor(self, number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    def floor_by_factor(self, number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    def smart_resize(self,
                     height: int,
                     width: int,
                     factor: int = 32,
                     min_pixels: int = 56 * 56,
                     max_pixels: int = 12845056) -> tuple[int, int]:
        """Smart resize image dimensions based on factor and pixel constraints"""
        h_bar = max(factor, self.round_by_factor(height, factor))
        w_bar = max(factor, self.round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = self.floor_by_factor(height / beta, factor)
            w_bar = self.floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, factor)
            w_bar = self.ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    def maybe_resize_bbox(self, left, top, right, bottom, img_width, img_height):
        """Resize bbox to ensure it's valid"""
        left = max(0, left)
        top = max(0, top)
        right = min(img_width, right)
        bottom = min(img_height, bottom)

        height = bottom - top
        width = right - left
        if height < 32 or width < 32:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = 32 / min(height, width)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)

            # Ensure the resized bbox is within image bounds
            new_left = max(0, new_left)
            new_top = max(0, new_top)
            new_right = min(img_width, new_right)
            new_bottom = min(img_height, new_bottom)

            new_height = new_bottom - new_top
            new_width = new_right - new_left

            if new_height > 32 and new_width > 32:
                return [new_left, new_top, new_right, new_bottom]
        return [left, top, right, bottom]


    def crop(self, image, bbox):
        img_width, img_height = image.size
        rel_x1, rel_y1, rel_x2, rel_y2 = bbox
        abs_x1, abs_y1, abs_x2, abs_y2 = rel_x1 / 1000. * img_width, rel_y1 / 1000. * img_height, rel_x2 / 1000. * img_width, rel_y2 / 1000. * img_height

        validated_bbox = self.maybe_resize_bbox(abs_x1, abs_y1, abs_x2, abs_y2, img_width, img_height)

        left, top, right, bottom = validated_bbox

        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))

        # Resize according to smart_resize logic
        new_w, new_h = self.smart_resize((right - left), (bottom - top), factor=32, min_pixels=256 * 32 * 32)
        cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
        return cropped_image, validated_bbox
