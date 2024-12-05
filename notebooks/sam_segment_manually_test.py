import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import cv2

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# Load the SAM model
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)


img_dir = Path("images/")
imgs = [f for f in img_dir.iterdir() if f.suffix.lower() == ".jpg"]

mask_dir = Path("spaceA-21_masks")
mask_dir.mkdir(exist_ok=True)

# Global variables for mouse callback
points = []
labels = []
img_display = None


def mouse_callback(event, x, y, flags, param):
    global points, labels, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        # Left click, label = 1 (interior point)
        points.append([x, y])
        labels.append(1)
        cv2.circle(img_display, (x, y), 3, (0, 255, 0), -1)  # Green dot for interior point
        cv2.imshow('image', img_display)
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click, label = 0 (exterior point)
        points.append([x, y])
        labels.append(0)
        cv2.circle(img_display, (x, y), 3, (0, 0, 255), -1)  # Red dot for exterior point
        cv2.imshow('image', img_display)

result_display_names = []
for img_f in imgs:
    # Read the image
    img = cv2.imread(str(img_f))
    img_display = img.copy()
    points = []
    labels = []
    cv2.namedWindow('image')
    cv2.imshow('image', img_display)
    cv2.setMouseCallback('image', mouse_callback)
    model_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(model_img)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key pressed
            if points:
                # Proceed to segmentation
                args = {
                    "point_coords": points,
                    "point_labels": labels,
                    "multimask_output": True
                    }
                print(f"points: {points}")
                print(f"labels: {labels}")
                masks, scores, logits = predictor.predict(
                    **args
                )

                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                logits = logits[sorted_ind]
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    color = (np.random.random(3) * 255).astype(np.uint8)
                    h, w = mask.shape[-2:]
                    mask = mask.astype(np.uint8)
                    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                    # borders
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    # try to smooth contours
                    contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                    mask_image = cv2.drawContours(mask_image, contours, -1, (255, 255, 255), thickness=2)
                    img_result = cv2.addWeighted(img_display, 0.6, mask_image, 0.4, 0)
                    result_display_names.append(f"result_{i}")
                    cv2.imshow(f'result_{i}', img_result)
                    cv2.setWindowTitle(f"result_{i}", f"result_{i}: score: {score}")

                # Wait for 's' or 'd' key
                while True:
                    key2 = cv2.waitKey(10) & 0xFF
                    if key2 == ord('s'):
                        # TODO: Save the mask and proceed to next image
                        cv2.imwrite(str(mask_dir / f"{img_f.stem}_mask.png"), mask)
                        for _name in result_display_names:
                            cv2.destroyWindow(_name)
                        cv2.destroyWindow('image')
                        break
                    elif key2 == ord('d'):
                        # Redo segmentation
                        for _name in result_display_names:
                            cv2.destroyWindow(_name)
                        img_display = img.copy()
                        points = []
                        labels = []
                        cv2.imshow('image', img_display)
                        break
                if key2 == 32:
                    break  # Proceed to next image
            else:
                print("No points selected. Please select points before pressing Enter.")
        elif key == ord('d'):
            # Redo current image segmentation
            img_display = img.copy()
            points = []
            labels = []
            cv2.imshow('image', img_display)
        elif key == 27:  # ESC key to exit
            cv2.destroyAllWindows()
            exit()
        elif key == 32:
            break

cv2.destroyAllWindows()
