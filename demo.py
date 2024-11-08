# -*- encoding: utf-8 -*-
# Time: 2024/11/7 18:37
# Auth: litre
# File: demo.py
# IDE: PyCharm
# @Contact: litre-wu@tutanota.com
import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import pathlib
import time


def main():
    img = 'ctrip_0.jpg'
    image = cv2.imread(img)
    # 创建掩膜
    masked_image = cv2.rectangle(np.zeros(image.shape[:2], dtype=np.uint8),
                                 (int(image.shape[1] * 0.7), int(image.shape[0] * 0.95)),
                                 (image.shape[1], image.shape[0]), (255), -1)
    cv2.imwrite('mask.png', masked_image)
    clock = time.time()
    inpainting = pipeline(Tasks.image_inpainting, model='model/cv_fft_inpainting_lama')
    print("模型加载:", time.time() - clock)
    clock = time.time()
    result = inpainting({
        'img': img,
        'mask': str(pathlib.Path(__file__).parent / 'mask.png'),
    })
    vis_img = result[OutputKeys.OUTPUT_IMG]
    print("预测:", time.time() - clock)
    cv2.imwrite('result.png', vis_img)


if __name__ == '__main__':
    main()
