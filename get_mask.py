import argparse
import cv2
import numpy as np
import mediapipe
import os


class PoseAware:
    def __init__(self):
        self.numbers = [234, 50, 36, 49, 45, 4, 275, 279, 266, 280, 454, 323, 361, 435, 288, 397, 365, 379, 378, 400, 377,
                        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234]
        self.pairs = [[self.numbers[i], self.numbers[i + 1]] for i in range(0, len(self.numbers) - 1)]
        self.mp_face_mesh = mediapipe.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True)

    def generate_mask(self, img, face_crop=False, image_size=(256, 256), crop_margin=0.1):
        routes = []
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        landmarks = results.multi_face_landmarks[0]

        if face_crop:
            img, landmarks = self.crop_face(img, landmarks, crop_margin)

        for source_idx, target_idx in self.pairs:
            source = landmarks.landmark[source_idx]
            target = landmarks.landmark[target_idx]
            relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
            relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))
            routes.append(relative_source)
            routes.append(relative_target)

        mask = self.create_mask(img, routes)
        img, mask = self.resize_images(img, mask, image_size)

        return img, mask

    def crop_face(self, img, landmarks, crop_margin):
        x_min = int(min([landmark.x for landmark in landmarks.landmark]) * img.shape[1])
        x_max = int(max([landmark.x for landmark in landmarks.landmark]) * img.shape[1])
        y_min = int(min([landmark.y for landmark in landmarks.landmark]) * img.shape[0])
        y_max = int(max([landmark.y for landmark in landmarks.landmark]) * img.shape[0])
        x_max = min(x_max + int(crop_margin * (x_max - x_min)), img.shape[1])
        x_min = max(x_min - int(crop_margin * (x_max - x_min)), 0)
        y_max = min(y_max + int(crop_margin * (y_max - y_min)), img.shape[0])
        y_min = max(y_min - int(crop_margin * (y_max - y_min)), 0)

        width = x_max - x_min
        height = y_max - y_min
        aspect_ratio = float(img.shape[1]) / float(img.shape[0])

        adjusted_width = width
        adjusted_height = int(adjusted_width / aspect_ratio)
        y_max = min(y_max, y_min + adjusted_height)
        img = img[y_min:y_max, x_min:x_max]

        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        landmarks = results.multi_face_landmarks[0]

        return img, landmarks

    def create_mask(self, img, routes):
        mask = np.zeros_like(img)
        polygon_points = np.array(routes, dtype=np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [polygon_points], (255, 255, 255))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
        return mask

    def resize_images(self, img, mask, image_size):
        img = cv2.resize(img, image_size)
        mask = cv2.resize(mask, image_size)
        return img, mask


def main(args):
    make_mask = PoseAware()
    video = cv2.VideoCapture(args.video_path)
    success, img = video.read()
    idx = 1

    while success:
        success, img = video.read()
        if not success:
            break
        image_size = (256, 256)
        crop_margin = 0.2
        img, mask = make_mask.generate_mask(img, face_crop=True, image_size=image_size, crop_margin=crop_margin)
        mask_name = f"{idx:05d}_front_mask.jpg"
        frame_name = f"{idx:05d}.jpg"
        os.makedirs(args.mask_output_dir, exist_ok=True)
        os.makedirs(args.frame_output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(args.mask_output_dir, mask_name), mask)
        cv2.imwrite(os.path.join(args.frame_output_dir, frame_name), img)
        idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate masks from a video using PoseAware.")
    parser.add_argument("--video_path", type=str, help="Path to the input video.")
    parser.add_argument("--mask_output_dir", type=str, default="./mask", help="Directory to save generated masks.")
    parser.add_argument("--frame_output_dir", type=str, default="./frame", help="Directory to save processed frames.")
    args = parser.parse_args()
    main(args)
