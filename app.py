import os
import cv2
from ultralytics import YOLO


class ParasiteDetector:
    def __init__(self, model_path, threshold=0.5):
        self.model = YOLO(model_path)
        self.threshold = threshold

    def detect_parasites(self, input_dir, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print("No images found in input directory.")
            return

        for image_file in image_files:
            image_path = os.path.join(input_dir, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read image: {image_file}")
                continue

            results = self.model.predict(source=image, conf=self.threshold, verbose=False)

            for result in results:
                boxes = result.boxes

                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])

                        label = f"{self.model.names[cls]} {conf:.2f}"

                        # Draw bounding box
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, image)
            print(f"Processed: {image_file} -> {output_path}")


if __name__ == "__main__":
    MODEL_PATH = "best.pt"          # your trained YOLO model
    INPUT_DIR = "input_images"      # folder with images
    OUTPUT_DIR = "output_images"    # folder to save results
    THRESHOLD = 0.5

    detector = ParasiteDetector(MODEL_PATH, THRESHOLD)
    detector.detect_parasites(INPUT_DIR, OUTPUT_DIR)
