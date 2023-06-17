import argparse
import cv2
import numpy as np
import torch
import facer
import os
from pathlib import Path
from insightface.app import FaceAnalysis
from ultralytics import YOLO


def find_face_for_recon(app, image):
    face_list = app.get(image)
    best_result = None
    best_area = 0
    for face in face_list:
        (left, top, right, bottom) = face.bbox
        area = (bottom - top) * (right - left)
        if best_result is None or area > best_area:
            best_result = face
            best_area = area

    if best_result is not None:
        recon = app.models['recognition']
        return recon.get(image, best_result)

    return None

def find_face_box_with_recon(app, image, target_emb, threshold=0.2):
    face_list = app.get(image)
    best_face = None
    best_ssim = 0

    if len(face_list) == 0:
        # face recon can fail if the face is too big, so assume the whole picture
        return (0, 0, image.shape[1], image.shape[0])

    if target_emb is None:
        return face_list[0].bbox

    recon = app.models['recognition']

    for face in face_list:
        emb = recon.get(image, face)
        ssim = recon.compute_sim(target_emb, emb)

        if ssim >= threshold and (best_face is None or ssim > best_ssim):
            best_face = face
            best_ssim = ssim

    return best_face.bbox if best_face is not None else None


def dilate_mask(mask, amount, blur_radius):
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (amount, amount))
    result = cv2.dilate(mask, dilate_kernel, iterations=1)
    result = cv2.GaussianBlur(result, (blur_radius, blur_radius), 0)
    return result


def mask_blend(src1, src2, mask):
    return src1 * mask + src2 * (1.0 - mask)


def find_nearest_face(faces, x, y):
    best_result = None
    best_dist = 0
    num_faces = faces["rects"].shape[0]
    for i in range(0, num_faces):
        (left, top, right, bottom) = faces["rects"][i]
        fx = (left + right) / 2.0
        fy = (top + bottom) / 2.0
        dist = (x - fx) ** 2 + (y - fy) ** 2

        if best_result is None or dist < best_dist:
            best_result = i
            best_dist = dist

    return best_result


def main(args):
    face_detector = facer.face_detector('retinaface/mobilenet', device=args.device)
    face_parser = facer.face_parser('farl/lapa/448', device=args.device)
    recon_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    recon_app.prepare(ctx_id=0, det_size=(640, 640))
    body_seg = YOLO('yolov8x-seg.pt')
    target_emb = None

    if args.target_ref and os.path.exists(args.target_ref):
        rf_image = cv2.imread(args.target_ref)
        target_emb = find_face_for_recon(recon_app, rf_image)

    src_files = sorted(os.listdir(args.input_dir))

    for sf in src_files:
        if Path(sf).suffix not in ['.png', '.jpg', '.jpeg', '.webp']:
            continue

        print('PROCESSING:', sf)

        sf_path = os.path.join(args.input_dir, sf)
        sf_name = Path(sf).stem

        out_name = sf_name + ".png"
        out_path = os.path.join(args.output_dir, out_name)

        if not args.overwrite and os.path.exists(out_path):
            print("Output file already exists; skipping.")
            continue

        image = facer.hwc2bchw(facer.read_hwc(sf_path)).to(device=args.device)

        with torch.inference_mode():
            face_box = find_face_box_with_recon(recon_app, cv2.imread(sf_path), target_emb)

            if face_box is None:
                print("Target face was not found, skipping.")
                continue

            face_cx = (face_box[0] + face_box[2]) / 2.0
            face_cy = (face_box[1] + face_box[3]) / 2.0

            faces = face_detector(image)

            if not faces:
                print("Failed to detect faces! Skipping")
                continue

            face_idx = find_nearest_face(faces, face_cx, face_cy)

            fparsed = face_parser(image, faces)
            seg_labels = fparsed["seg"]["label_names"]
            seg_logits = fparsed["seg"]["logits"]
            seg_probs = seg_logits.softmax(dim=1)
            full_mask = seg_probs.argmax(dim=1)  # .float() / seg_probs.size(1)
            full_mask = torch.squeeze(full_mask, 0).numpy()

            # check for multiple faces
            if len(full_mask.shape) > 2:
                full_mask = np.squeeze(full_mask[face_idx, ...])

            face_mask = full_mask.copy()
            face_mask[face_mask == seg_labels.index('hair')] = 0
            face_mask[face_mask > 0] = 1
            face_mask = face_mask.astype(np.float32)

            hair_mask = full_mask.copy()
            hair_mask[hair_mask != seg_labels.index('hair')] = 0
            hair_mask[hair_mask > 0] = 1
            hair_mask = hair_mask.astype(np.float32)

            result = body_seg(sf_path, device=args.device, classes=[0], retina_masks=True)

            if result[0].masks is None:
                print('Unable to segment image; skipping')
                continue

            # match face to body by finding body mask that overlaps the most with face mask
            best_score = 0
            body_mask = None
            for r in result[0].masks.data:
                m = r.numpy()
                score = np.sum(face_mask * m)
                if body_mask is None or score > best_score:
                    best_score = score
                    body_mask = m

            # put everything together
            composed = np.full(body_mask.shape, args.bg_weight)
            composed = mask_blend(np.full(composed.shape, args.body_weight), composed, dilate_mask(body_mask, 5, 3))
            composed = mask_blend(np.full(composed.shape, args.hair_weight), composed, dilate_mask(hair_mask, 15, 13))
            composed = mask_blend(np.full(composed.shape, args.face_weight), composed, dilate_mask(face_mask, 9, 7))

            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            cv2.imwrite(out_path, (composed*255).astype(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Person Mask Generator')
    parser.add_argument('--device', type=str, default='cpu', help='Name of torch device for inference')
    parser.add_argument('--input-dir', type=str, help='Path to folder with input images')
    parser.add_argument('--output-dir', type=str, help='Path to output folder, it will be created if missing')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite mask if file already exists in output folder')
    parser.add_argument('--target-ref', type=str, help='Optional. Path to an image of a face for recognizing correct subject when multiple candidates exist in the same image.')
    parser.add_argument('--face-weight', type=float, default=1.0, help='Weight to use for face pixels, must be a value between 0 and 1')
    parser.add_argument('--hair-weight', type=float, default=0.6, help='Weight to use for hair pixels, must be a value between 0 and 1')
    parser.add_argument('--body-weight', type=float, default=0.25, help='Weight to use for body pixels, must be a value between 0 and 1')
    parser.add_argument('--bg-weight', type=float, default=0.0, help='Weight to use for non-subject pixels, must be a value between 0 and 1')
    args = parser.parse_args()
    main(args)
