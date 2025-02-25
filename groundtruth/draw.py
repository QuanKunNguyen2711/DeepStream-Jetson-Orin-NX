import cv2
import json
import argparse
import numpy as np

# DeepStream definitions (absolute pixel coordinates on 1920x1080)
ROI = [(286, 859), (610, 830), (400, 672), (215, 680)]
STOP_LINES = {
    "1": (286, 859, 403, 848),
    "2": (403, 848, 494, 840),
    "3": (496, 840, 610, 830)
}
DIRECTION_LINES = {
    "right": (161, 948, 225, 1081), 
    "left": (716, 822, 1136, 1074),
    "u-turn": (600, 823, 787, 810),
    "straight": (360, 1077, 955, 1075)
}


def load_annotations(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def draw_deepstream_overlays(img):
    pts = np.array(ROI, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    
    for key, (x1, y1, x2, y2) in STOP_LINES.items():
        cv2.line(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        cv2.putText(img, f"Stop {key}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    for key, (x1, y1, x2, y2) in DIRECTION_LINES.items():
        cv2.line(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(img, key, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

def draw_label_studio_annotations(img, annotations, frame_no, img_width, img_height, scale=1.0):
    for entry in annotations:
        for ann in entry.get("annotations", []):
            for result in ann.get("result", []):
                if result.get("type") != "videorectangle":
                    continue
                value = result.get("value", {})
                sequence = value.get("sequence", [])
                label = value.get("labels", [""])[0]
                for item in sequence:
                    if item.get("frame", 0) == frame_no:
                        x_pct = item.get("x", 0) * scale
                        y_pct = item.get("y", 0) * scale
                        width_pct = item.get("width", 0) * scale
                        height_pct = item.get("height", 0) * scale
                        # Convert percentage values to absolute pixel coordinates
                        x_abs = int((x_pct / 100) * img_width)
                        y_abs = int((y_pct / 100) * img_height)
                        w_abs = int((width_pct / 100) * img_width)
                        h_abs = int((height_pct / 100) * img_height)

                        # Use different colors for enabled/disabled boxes
                        color = (0, 0, 255) if item.get("enabled", True) else (0, 255, 255)
                        cv2.rectangle(img, (x_abs, y_abs), (x_abs+w_abs, y_abs+h_abs), color, 2)
                        cv2.putText(img, label, (x_abs, y_abs-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        break

def main():
    # args = parse_args()
    json_input = "/home/quannguyen/nota/groundtruth/exported_annotations.json"  
    frame = 1  

    img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    img_width, img_height = 1920, 1080

    annotations = load_annotations(json_input)
    
    draw_deepstream_overlays(img)
    
    draw_label_studio_annotations(img, annotations, frame, img_width, img_height)

    output_filename = "output.png"
    cv2.imwrite(output_filename, img)
    print(f"Visualization saved as {output_filename}")

if __name__ == "__main__":
    main()
