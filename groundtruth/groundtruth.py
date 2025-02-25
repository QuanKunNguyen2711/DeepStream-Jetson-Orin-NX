import json
import math
import csv

# Configuration (same as DeepStream setup TranHungDao)
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
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

def point_line_distance(px, py, x1, y1, x2, y2):
    line_len_sq = (x2 - x1)**2 + (y2 - y1)**2
    if line_len_sq == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1)*(x2 - x1) + (py - y1)*(y2 - y1)) / line_len_sq
    t = max(0, min(1, t))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)
    return math.hypot(px - proj_x, py - proj_y)

def segments_intersect(a, b, c, d):
    def ccw(A, B, C):
        return (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])
    ccw1 = ccw(a, b, c)
    ccw2 = ccw(a, b, d)
    if ccw1 * ccw2 > 0:
        return False
    ccw3 = ccw(c, d, a)
    ccw4 = ccw(c, d, b)
    return ccw3 * ccw4 <= 0

def is_inside_roi(x, y):
    n = len(ROI)
    inside = False
    p1x, p1y = ROI[0]
    for i in range(n + 1):
        p2x, p2y = ROI[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_inters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def process_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    ground_truth = []
    
    for entry in data:
        for ann in entry.get('annotations', []):
            for result in ann.get('result', []):
                if result.get('type') != 'videorectangle':
                    continue
                value = result.get('value', {})
                sequence = sorted(value.get('sequence', []), key=lambda x: x['frame'])
                labels = value.get('labels', ['unknown'])
                vehicle_type = labels[0]
                
                path = []
                for frame in sequence:
                    if not frame.get('enabled', True):
                        continue
                    x = (frame['x'] + frame['width'] / 2) * IMG_WIDTH / 100
                    y = (frame['y'] + frame['height']) * IMG_HEIGHT / 100
                    path.append((x, y))
                
                lane, direction = None, None
                prev = None
                for coord in path:
                    x, y = coord
                    if not is_inside_roi(x, y):
                        prev = coord
                        continue
                    
                    # Detect stop line crossing
                    if not lane:
                        closest = None
                        min_dist = float('inf')
                        for lid, line in STOP_LINES.items():
                            dist = point_line_distance(x, y, *line)
                            if dist < 5 and dist < min_dist:
                                closest = lid
                                min_dist = dist
                        if prev and not closest:
                            for lid, line in STOP_LINES.items():
                                x1, y1, x2, y2 = line
                                if segments_intersect(prev, coord, (x1, y1), (x2, y2)):
                                    closest = lid
                                    break
                        if closest:
                            lane = closest
                    
                    # Detect direction line crossing
                    if lane and not direction:
                        closest_dir = None
                        min_dist_dir = float('inf')
                        for did, line in DIRECTION_LINES.items():
                            dist = point_line_distance(x, y, *line)
                            if dist < 5 and dist < min_dist_dir:
                                closest_dir = did
                                min_dist_dir = dist
                        if prev and not closest_dir:
                            for did, line in DIRECTION_LINES.items():
                                x1, y1, x2, y2 = line
                                if segments_intersect(prev, coord, (x1, y1), (x2, y2)):
                                    closest_dir = did
                                    break
                        if closest_dir:
                            direction = closest_dir
                            break
                    
                    prev = coord
                
                if lane and direction:
                    ground_truth.append({
                        'vehicle_type': vehicle_type,
                        'lane': lane,
                        'direction': direction
                    })
    
    with open('ground_truth.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['vehicle_type', 'lane', 'direction'])
        writer.writeheader()
        writer.writerows(ground_truth)

process_annotations('exported_annotations.json')