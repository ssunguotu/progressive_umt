import imgaug.augmenters as iaa
import imgaug as ia
import cv2
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np

image = cv2.imread('./image.jpg')

bbxes = []
gt_box = [525, 581, 662, 650, '0']
# cv2.rectangle(image, gt_box[0:2], gt_box[2:4], (204, 0, 0), 2)


def bbs2numpy(bbs):
    bboxes = []
    for bb in bbs.bounding_boxes:
        x1 = bb.x1 - 1
        y1 = bb.y1 - 1
        w = bb.x2 - bb.x1
        h = bb.y2 - bb.y1
        label = float(bb.label)
        bboxes.append([x1, y1, w, h, label])
    return np.array(bboxes, dtype=np.int32)

x1 = gt_box[0] + 1
y1 = gt_box[1] + 1
x2 = gt_box[2]
y2 = gt_box[3]
bbxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=str(gt_box[4])))
bbs = BoundingBoxesOnImage(bbxes, shape=image.shape)

seq = iaa.Sequential([
        # small object: 剪裁 + 旋转(注意padding不要进行复制/镜像，会使得存在一些多余的object)
        # Dropout
        # iaa.CropAndPad(
        #     percent=(0.1, 0.3),
        #     pad_mode=['constant', 'edge', 'linear_ramp', 'maximum', 'median',
        #               'minimum', 'constant', 'linear_ramp'],
        #     pad_cval=(0, 255)
        # ),
        iaa.Affine(rotate=(-45, 45), mode=['constant', 'edge']),
        # iaa.CoarseDropout(0.02, size_percent=0.5)
        iaa.MotionBlur(k=5, angle=[-45, 45])
    ])

aug_image, bbs_aug = seq(image=image, bounding_boxes=bbs)

bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
gt_boxes = bbs2numpy(bbs_aug).squeeze(0)
x1 = gt_boxes[0] + 1
y1 = gt_boxes[1] + 1
x2 = gt_boxes[2] + x1
y2 = gt_boxes[3] + y1
print(gt_boxes)
cv2.rectangle(aug_image, (x1,y1), (x2,y2), (204, 0, 204), 2)

# origin
cv2.rectangle(image, gt_box[0:2], gt_box[2:4], (204, 0, 0), 2)

cv2.imwrite('./aug.jpg', aug_image)
cv2.imwrite('./aug_before.jpg', image)