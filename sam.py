import numpy as np
import os
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms as T
from pathlib import Path
import sys
from segment_anything import sam_model_registry, SamPredictor

def get_one_box(det_output, thrd=0.9):
    max_area = 0
    max_bbox = None

    if det_output['boxes'].shape[0] == 0 or thrd < 1e-5:
        return None

    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        if float(score) < thrd:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if float(area) > max_area:
            max_bbox = [float(x) for x in bbox]
            max_area = area

    if max_bbox is None:
        return get_one_box(det_output, thrd=thrd - 0.1)

    return max_bbox

output_dir = './demo/blender'
mask_dir = './demo/mask'
folder_path = Path(output_dir)

img_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

for image_path in img_paths:
    filename = os.path.splitext(os.path.basename(image_path))[0]

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    det_model = fasterrcnn_resnet50_fpn(pretrained=True).cuda().eval()
    det_transform = T.Compose([T.ToTensor()])
    det_input = det_transform(image).cuda()
    det_output = det_model([det_input])[0]
    bbox = get_one_box(det_output)  # xyxy
    #bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]  # xywh
    input_box = np.array(bbox)

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )

    transparent = (masks[0] * 255).astype(np.uint8)
    cv2.imwrite(f'{mask_dir}/{filename}_mask.png', transparent)
