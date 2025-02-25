#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
# https://youtu.be/MlFTwvwe0xw
import sys

sys.path.append("../")
import gi
import configparser
import argparse

gi.require_version("Gst", "1.0")
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst
from gi.repository import GLib, GstRtspServer
from ctypes import *
import time
import csv
from datetime import datetime
import sys
import os
import math
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import PERF_DATA

import pyds
import cv2
import numpy as np
from collections import defaultdict
import threading
from queue import Queue
import threading

no_display = False
silent = False
file_loop = False
perf_data = None

MAX_DISPLAY_LEN = 64
PGIE_CLASS_ID_MOTORBIKE = 3
PGIE_CLASS_ID_CAR = 2
PGIE_CLASS_ID_TRUCK = 5
PGIE_CLASS_ID_BUS = 1
PGIE_CLASS_ID_VAN = 6
PGIE_CLASS_ID_BICYCLE = 0
PGIE_CLASS_ID_PEDESTRIAN = 4
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080  # 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 640  # 1280
TILED_OUTPUT_HEIGHT = 360  # 720
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 0
OSD_DISPLAY_TEXT = 1

pgie_classes_str = ["bicycle", "bus", "car", "motorbike", "pedestrian", "truck", "van"]

STREAMS_NAME = {
    "stream0": "DongKhoi_MacThiBuoi",
    "stream1": "RachBungBinh_NguyenThong_1",
    "stream2": "TranHungDao_NguyenVanCu",
    "stream3": "TranKhacChan_TranQuangKhai"
}

ROIS = {
    "stream0": [(201, 939), (840, 870), (365, 656), (65, 665)],
    "stream1": [(497, 861), (814, 834), (600, 542), (522, 546)],
    "stream2": [(286, 859), (610, 830), (400, 672), (215, 680)],
    "stream3": [(641, 590), (972, 558), (666, 137), (606, 142)]
}

ROI_ARRAYS = {
    stream_id: np.array(points, dtype=np.int32) 
    for stream_id, points in ROIS.items()
}

STOP_LINES = {
    # DongKhoi_MacThiBuoi
    "stream0": {
        "1": (201, 939, 437, 908),
        "2": (437, 908, 658, 884),
        "3": (658, 884, 840, 870)
    },
    # RachBungBinh_NguyenThong_1
    "stream1": {
        "1": (497, 861, 814, 834),
    },
    # TranHungDao_NguyenVanCu
    "stream2": {
        "1": (286, 859, 403, 848),
        "2": (403, 848, 494, 840),
        "3": (496, 840, 610, 830)
    },
    # TranKhacChan_TranQuangKhai
    "stream3": {
        "1": (641, 590, 972, 558),
    },
}

DIRECTION_LINES = {
    "stream0": {
        "right": (192, 1003, 256, 1078), 
        "left": (1138, 943, 1572, 1088),
        "straight": (366, 1072, 1430, 1072)
    },
    "stream1": {
        "right": (401, 960, 455, 1078),
        "left": (1269, 894, 1559, 1076),
        "u-turn": (816, 830, 1100, 810),
        "straight": (528, 1076, 1199, 1072)
    },
    "stream2": {
        "right": (161, 948, 225, 1081), 
        "left": (716, 822, 1136, 1074),
        "u-turn": (600, 823, 787, 810),
        "straight": (360, 1077, 955, 1075)
    },
    "stream3": {
        "right": (442, 914, 522, 1074), 
        "left": (1518, 679, 1633, 1130),
        "u-turn": (974, 556, 1192, 530),
        "straight": (600, 1070, 1637, 1074)
    }
}

    
def draw_lines(frame_meta, lines_config, is_stop_lines):
    display_meta = pyds.nvds_acquire_display_meta_from_pool(frame_meta.base_meta.batch_meta)
    display_meta.num_lines = len(lines_config)
    
    colors = {
        "right": (1.0, 0.0, 0.0, 1.0),
        "left": (0.0, 1.0, 0.0, 1.0),
        "u-turn": (0.0, 0.0, 1.0, 1.0),
        "straight": (1.0, 1.0, 0.0, 1.0),
    } if not is_stop_lines else {
        "1": (1.0, 0.0, 0.0, 1.0),
        "2": (0.0, 1.0, 0.0, 1.0),
        "3": (0.0, 0.0, 1.0, 1.0),
        "4": (1.0, 1.0, 0.0, 1.0),
        "5": (1.0, 0.0, 1.0, 1.0),
        "6": (0.0, 1.0, 1.0, 1.0)
    }
    
    for i, (line_id, (x1, y1, x2, y2)) in enumerate(lines_config.items()):
        line_params = display_meta.line_params[i]
        line_params.x1, line_params.y1 = x1, y1
        line_params.x2, line_params.y2 = x2, y2
        line_params.line_width = 2
        
        color = colors.get(line_id, (1.0, 1.0, 1.0, 1.0))
        line_params.line_color.set(*color)

    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
    

def draw_rois(frame_meta, rois, stream_id):
    if stream_id not in rois:
        return

    display_meta = pyds.nvds_acquire_display_meta_from_pool(frame_meta.base_meta.batch_meta)
    display_meta.num_lines = len(rois[stream_id])  

    color = (1.0, 1.0, 1.0, 1.0)  

    roi_points = rois[stream_id]

    for i in range(len(roi_points)):
        x1, y1 = roi_points[i]
        x2, y2 = roi_points[(i + 1) % len(roi_points)]
        line_params = display_meta.line_params[i]
        line_params.x1, line_params.y1 = x1, y1
        line_params.x2, line_params.y2 = x2, y2
        line_params.line_width = 3
        line_params.line_color.set(*color)

    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)


def write_traffic_volume(stream_id, obj_id, vehicle):
    csv_file = f"{STREAMS_NAME[stream_id]}.csv"
    
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(['date', 'entry_time', 'stop_line_time', 'direction_time', 'vehicle_type', 'vehicle_id', 'lane', 'direction'])
        
        writer.writerow([
            vehicle.get('date', ''),
            vehicle.get('entry_time', ''),
            vehicle.get('stop_line_time', ''),
            vehicle.get('direction_time', ''),
            vehicle.get('vehicle_type', ''),
            obj_id,
            vehicle.get('lane_id', ''),
            vehicle.get('direction', '')
        ])
  
  
def point_line_distance(px, py, x1, y1, x2, y2):
    line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if line_len_sq == 0:
        return math.hypot(px - x1, py - y1)

    t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq
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
    if ccw3 * ccw4 > 0:
        return False

    return True


def check_line_crossing(coordinate, lines, threshold=5):
    x_center = coordinate[0]
    y_bottom = coordinate[1]
    closest = None
    min_distance = float('inf')
    for lane_id, (x1, y1, x2, y2) in lines.items():
        distance = point_line_distance(x_center, y_bottom, x1, y1, x2, y2)
        if distance < threshold and distance < min_distance:
            closest = lane_id
            min_distance = distance

    return closest


def is_inside_roi(obj_meta, roi_points):
    roi_array = np.array(roi_points, dtype=np.int32)
    
    rect = obj_meta.rect_params
    cx = int(rect.left + rect.width/2)
    cy = int(rect.top + rect.height/2)
    
    return cv2.pointPolygonTest(roi_array, (cx, cy), False) >= 0


class TrackedVehicles:
    def __init__(self):
        self.data = defaultdict(dict)
        self.locks = defaultdict(threading.Lock)
        self.last_cleanup = defaultdict(int)

    def add_vehicle(self, stream_id, obj_id, entry):
        with self.locks[stream_id]:
            entry['xy_history'] = [entry['xy']]
            self.data[stream_id][obj_id] = entry

    def update_vehicle(self, stream_id, obj_id, coordinate, new_last_seen):
        with self.locks[stream_id]:
            if obj_id in self.data[stream_id]:
                vehicle = self.data[stream_id][obj_id]
                vehicle['xy'] = coordinate
                vehicle['xy_history'].append(coordinate)

                if len(vehicle['xy_history']) > 5:
                    vehicle['xy_history'].pop(0)

                vehicle['last_seen'] = new_last_seen


    def remove_vehicle(self, stream_id, obj_id):
        with self.locks[stream_id]:
            if obj_id in self.data[stream_id]:
                del self.data[stream_id][obj_id]

    def cleanup_expired(self, stream_id, current_frame, max_age=30):
        if current_frame - self.last_cleanup[stream_id] < 30:
            return
        
        with self.locks[stream_id]:
            self.last_cleanup[stream_id] = current_frame
            expired_ids = [
                obj_id 
                for obj_id, vehicle in self.data[stream_id].items()
                if current_frame - vehicle['last_seen'] > max_age
            ]

            for obj_id in expired_ids:
                del self.data[stream_id][obj_id]



tracked_vehicles = TrackedVehicles()

def nvtracker_src_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            stream_id = f"stream{frame_meta.pad_index}"
            current_frame_num = frame_meta.frame_num

            # if stream_id in STOP_LINES:
            #     draw_lines(frame_meta, STOP_LINES[stream_id], is_stop_lines=True)
            # if stream_id in DIRECTION_LINES:
            #     draw_lines(frame_meta, DIRECTION_LINES[stream_id], is_stop_lines=False)
            # if stream_id in ROIS:
            #     draw_rois(frame_meta, ROIS, stream_id)
                
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        obj_counter = {
            PGIE_CLASS_ID_MOTORBIKE: 0,
            PGIE_CLASS_ID_CAR: 0,
            PGIE_CLASS_ID_TRUCK: 0,
            PGIE_CLASS_ID_BUS: 0,
            PGIE_CLASS_ID_VAN: 0,
            PGIE_CLASS_ID_BICYCLE: 0,
            PGIE_CLASS_ID_PEDESTRIAN: 0
        }
        
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                     
            except StopIteration:
                break
            
            obj_id = obj_meta.object_id
            x_center = obj_meta.rect_params.left + obj_meta.rect_params.width / 2
            y_bottom = obj_meta.rect_params.top + obj_meta.rect_params.height
            current_frame = frame_number
            
            # ROI detection
            roi_array = ROI_ARRAYS.get(stream_id)
            is_in_roi = roi_array is not None and is_inside_roi(obj_meta, roi_array)
            
            if is_in_roi:
                if obj_id not in tracked_vehicles.data[stream_id]:
                    entry = {
                        'date': datetime.now().strftime("%Y-%m-%d"),
                        'entry_time': datetime.now().strftime("%H:%M:%S"),
                        'stop_line_time': None,
                        'direction': None,
                        'direction_time': None,
                        'last_seen': current_frame,
                        'vehicle_type': pgie_classes_str[obj_meta.class_id],
                        'lane_id': None,
                        'xy': (x_center, y_bottom)
                    }
                    tracked_vehicles.add_vehicle(stream_id, obj_id, entry)
                    # print(f"{stream_id};{current_frame};{entry['entry_time']}: Add {pgie_classes_str[obj_meta.class_id]} {obj_id}")
                    obj_counter[obj_meta.class_id] += 1
                    
            if obj_id in tracked_vehicles.data[stream_id]:
                vehicle = tracked_vehicles.data[stream_id][obj_id]
                
                tracked_vehicles.update_vehicle(stream_id, obj_id, (x_center, y_bottom), current_frame)
                if not vehicle['stop_line_time']:
                    lane_id = check_line_crossing((x_center, y_bottom), STOP_LINES.get(stream_id, {}))
                    if not lane_id:
                        for i in range(len(vehicle['xy_history']) - 1):
                            prev = vehicle['xy_history'][i]
                            curr = vehicle['xy_history'][i+1]
                            for stop_lane_id, line in STOP_LINES.get(stream_id, {}).items():
                                x1, y1, x2, y2 = line
                                if segments_intersect(prev, curr, (x1, y1), (x2, y2)):
                                    lane_id = stop_lane_id
                                    break
                            if lane_id:
                                break

                    if lane_id:
                        vehicle['stop_line_time'] = datetime.now().strftime("%H:%M:%S")
                        vehicle['lane_id'] = lane_id
                        vehicle['stop_line_frame'] = current_frame
                        # print(f"{stream_id};{current_frame}:{vehicle['stop_line_time']}: Update lane {lane_id} {pgie_classes_str[obj_meta.class_id]} {obj_id}")
                
                # Update the last stop line crossing
                elif vehicle['stop_line_time'] and vehicle['lane_id']:
                    lane_id = check_line_crossing((x_center, y_bottom), STOP_LINES.get(stream_id, {}))
                    if lane_id is not None and lane_id != vehicle['lane_id']:
                        vehicle['stop_line_time'] = datetime.now().strftime("%H:%M:%S")
                        vehicle['lane_id'] = lane_id
                        vehicle['stop_line_frame'] = current_frame
                        # print(f"{stream_id};{current_frame}:{vehicle['stop_line_time']}: Update again lane {lane_id } {pgie_classes_str[obj_meta.class_id]} {obj_id}")
                    
                if vehicle['stop_line_time'] and not vehicle['direction']:
                    direction = check_line_crossing((x_center, y_bottom), DIRECTION_LINES.get(stream_id, {}))
                    if not direction:
                        for i in range(len(vehicle['xy_history']) - 1):
                            prev = vehicle['xy_history'][i]
                            curr = vehicle['xy_history'][i+1]
                            for direction_id, line in DIRECTION_LINES.get(stream_id, {}).items():
                                x1, y1, x2, y2 = line
                                if segments_intersect(prev, curr, (x1, y1), (x2, y2)):
                                    direction = direction_id
                                    break
                            if direction:
                                break
                            
                    if direction:
                        vehicle['direction'] = direction
                        vehicle['direction_time'] = datetime.now().strftime("%H:%M:%S")
                        # print(f"{stream_id};{current_frame}: Update direction {vehicle['vehicle_type']} {obj_id} {direction}")
                        write_traffic_volume(stream_id, obj_id, vehicle)
                        tracked_vehicles.remove_vehicle(stream_id, obj_id)
                            

            class_colors = {
                PGIE_CLASS_ID_MOTORBIKE: (0.0, 1.0, 0.0, 0.8),  # Green
                PGIE_CLASS_ID_CAR: (1.0, 0.0, 0.0, 0.8),        # Red
                PGIE_CLASS_ID_TRUCK: (0.0, 0.0, 1.0, 0.8),      # Blue
                PGIE_CLASS_ID_BUS: (1.0, 1.0, 0.0, 0.8),        # Yellow
                PGIE_CLASS_ID_VAN: (1.0, 0.0, 1.0, 0.8),        # Magenta
                PGIE_CLASS_ID_BICYCLE: (1.0, 0.5, 0.0, 0.8),    # Orange
                PGIE_CLASS_ID_PEDESTRIAN: (0.0, 1.0, 1.0, 0.8)  # Cyan
            }

            # Set bounding box color based on class_id
            if obj_meta.class_id in class_colors:
                obj_meta.rect_params.border_color.set(*class_colors[obj_meta.class_id])
                
            
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
            
        tracked_vehicles.cleanup_expired(stream_id, current_frame_num)
        # Update frame rate through this probe
        global perf_data
        perf_data.update_fps(stream_id)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    """
    The function is called when a new pad is created by the decodebin. 
    The function checks if the new pad is for video and not audio. 
    If the new pad is for video, the function checks if the pad caps contain NVMM memory features. 
    If the pad caps contain NVMM memory features, the function links the decodebin pad to the source bin
    ghost pad. 
    If the pad caps do not contain NVMM memory features, the function prints an error message.
    :param decodebin: The decodebin element that is creating the new pad
    :param decoder_src_pad: The source pad created by the decodebin element
    :param data: This is the data that was passed to the callback function. In this case, it is the
    source_bin
    """
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write(
                    "Failed to link decoder src pad to source bin ghost pad\n"
                )
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    """
    If the child added to the decodebin is another decodebin, connect to its child-added signal. If the
    child added is a source, set its drop-on-latency property to True.
    
    :param child_proxy: The child element that was added to the decodebin
    :param Object: The object that emitted the signal
    :param name: The name of the element that was added
    :param user_data: This is a pointer to the data that you want to pass to the callback function
    """
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property("drop-on-latency") != None:
            Object.set_property("drop-on-latency", True)


def create_source_bin(index, uri):
    """
    It creates a GstBin, adds a uridecodebin to it, and connects the uridecodebin's pad-added signal to
    a callback function
    
    :param index: The index of the source bin
    :param uri: The URI of the video file to be played
    :return: A bin with a uri decode bin and a ghost pad.
    """
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def make_element(element_name, i):
    """
    Creates a Gstreamer element with unique name
    Unique name is created by adding element type and index e.g. `element_name-i`
    Unique name is essential for all the element in pipeline otherwise gstreamer will throw exception.
    :param element_name: The name of the element to create
    :param i: the index of the element in the pipeline
    :return: A Gst.Element object
    """
    element = Gst.ElementFactory.make(element_name, element_name)
    if not element:
        sys.stderr.write(" Unable to create {0}".format(element_name))
    element.set_property("name", "{0}-{1}".format(element_name, str(i)))
    return element


def main(args, requested_pgie=None, config=None, disable_probe=False):
    input_sources = args
    number_sources = len(input_sources)
    global perf_data
    perf_data = PERF_DATA(number_sources)

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    print("Creating streamux \n ")
    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    pipeline.add(streammux)
    
    for i in range(number_sources):
        print("Creating source_bin ", i, " \n ")
        uri_name = input_sources[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)

    queue1 = Gst.ElementFactory.make("queue", "queue1")
    pipeline.add(queue1)
    
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    nvstreamdemux = Gst.ElementFactory.make("nvstreamdemux", "nvstreamdemux")

    streammux.set_property('width', MUXER_OUTPUT_WIDTH)
    streammux.set_property('height', MUXER_OUTPUT_HEIGHT)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
        
    streammux.set_property("batch-size", number_sources)
    pgie.set_property("config-file-path", "myds_pgie_config.txt")
    # pgie_batch_size = pgie.get_property("batch-size")
    pgie.set_property("batch-size", number_sources)
    
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    pipeline.add(tracker)
    #Set properties of tracker
    config = configparser.ConfigParser()
    config.read('myds_tracker_config.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame' :
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)

    pipeline.add(pgie)
    pipeline.add(nvstreamdemux)
    
    # linking
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(tracker)
    tracker.link(nvstreamdemux)
    ##creating demux src
    
    # RTSP Server Setup
    rtsp_port_num = 8554
    server = GstRtspServer.RTSPServer.new()
    server.props.service = str(rtsp_port_num)
    server.attach(None)

    for i in range(number_sources):
        # nvstreamdemux -> queue -> nvvidconv -> nvosd -> (if Jetson) nvegltransform -> nveglgl
        # Creating EGLsink
        if is_aarch64():
            print("Creating nv3dsink \n")
            sink = make_element("nv3dsink", i)
            if not sink:
                sys.stderr.write(" Unable to create nv3dsink \n")
        else:
            print("Creating EGLSink \n")
            sink = make_element("nveglglessink", i)

        sink.set_property('sync', 0)  # Disable sync to avoid waiting for clock
        sink.set_property('async', 0)  # Disable async to avoid internal queues
        sink.set_property('qos', 0)  # Disable QoS to prevent buffer drops due to deadlines
        sink = Gst.ElementFactory.make("fakesink", f"fake-video-renderer_{i}")
        pipeline.add(sink)

        # creating queue
        queue = make_element("queue", i)
        pipeline.add(queue)

        # creating nvvidconv
        nvvideoconvert = make_element("nvvideoconvert", i)
        pipeline.add(nvvideoconvert)

        # creating nvosd
        nvdsosd = make_element("nvdsosd", i)
        pipeline.add(nvdsosd)
        nvdsosd.set_property("process-mode", OSD_PROCESS_MODE)
        nvdsosd.set_property("display-text", OSD_DISPLAY_TEXT)

        # connect nvstreamdemux -> queue
        padname = "src_%u" % i
        demuxsrcpad = nvstreamdemux.get_request_pad(padname)

        queuesinkpad = queue.get_static_pad("sink")
        demuxsrcpad.link(queuesinkpad)

        # connect  queue -> nvvidconv -> nvosd -> nveglgl
        queue.link(nvvideoconvert)
        nvvideoconvert.link(nvdsosd)
        nvdsosd.link(sink)
        
        # # nvosd -> nvvidconv_postosd -> caps -> encoder -> rtppay -> udpsink
        # nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", f"convertor_postosd_{i}")
        # pipeline.add(nvvidconv_postosd)
        
        # # Create a caps filter
        # caps = Gst.ElementFactory.make("capsfilter", f"filter_{i}")
        # pipeline.add(caps)
        # caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
        
        # encoder = Gst.ElementFactory.make("nvv4l2h264enc", f"encoder_{i}")
        # pipeline.add(encoder)
        # encoder.set_property('bitrate', 4000000)
        
        # if is_aarch64():
        #     encoder.set_property('preset-level', 1)
        #     encoder.set_property('insert-sps-pps', 1)
                
        # # Make the payload-encode video into RTP packets
        # rtppay = Gst.ElementFactory.make("rtph264pay", f"rtppay_{i}")
        # pipeline.add(rtppay)
        
        # # Make the UDP sink
        # updsink_port_num = 5400 + i
        # udpsink = Gst.ElementFactory.make("udpsink", f"udpsink_{i}")
        # pipeline.add(udpsink)
        
        # udpsink.set_property('host', '224.224.255.255')
        # udpsink.set_property('port', updsink_port_num)
        # udpsink.set_property('async', 0)
        # udpsink.set_property('sync', 0)
        
        # # Link elements
        # nvdsosd.link(nvvidconv_postosd)
        # nvvidconv_postosd.link(caps)
        # caps.link(encoder)
        # encoder.link(rtppay)
        # rtppay.link(udpsink)

        # # Create RTSP Mount Point
        # factory = GstRtspServer.RTSPMediaFactory.new()
        # factory.set_launch(f"( udpsrc name=pay0 port={5400 + i} buffer-size=524288 "
        #                   f"caps=\"application/x-rtp, media=video, clock-rate=90000, "
        #                   f"encoding-name=H264, payload=96 \" )")
        # factory.set_shared(True)
        # server.get_mount_points().add_factory(f"/ds{i}", factory)
        # print(f"RTSP Stream available at rtsp://localhost:{rtsp_port_num}/ds{i}")
    

    print("Linking elements in the Pipeline \n")
    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    
    tracker_src_pad = tracker.get_static_pad("src")
    
    if not tracker_src_pad:
        sys.stderr.write(" Unable to get tracker_src_pad \n")
    else:
        tracker_src_pad.add_probe(Gst.PadProbeType.BUFFER, nvtracker_src_pad_buffer_probe, 0)
        # perf callback function to print fps every 5 sec
        GLib.timeout_add(5000, perf_data.perf_print_callback)


    print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    # stream_paths = parse_args()
    # stream_paths = [
    #     "file:///home/nvidia/deepstream_python_apps/apps/my-deepstream/videos/trimmed_DongKhoi_MacThiBuoi.h264",
    #     "file:///home/nvidia/deepstream_python_apps/apps/my-deepstream/videos/trimmed_RachBungBinh_NguyenThong_1.h264",
    #     "file:///home/nvidia/deepstream_python_apps/apps/my-deepstream/videos/trimmed_TranHungDao_NguyenVanCu.h264",
    #     "file:///home/nvidia/deepstream_python_apps/apps/my-deepstream/videos/trimmed_TranKhacChan_TranQuangKhai.h264"
    # ]
    stream_paths = [
        "file:///home/nvidia/Videos/trimmed_videos/trimmed_DongKhoi_MacThiBuoi.h264",
        "file:///home/nvidia/Videos/trimmed_videos/trimmed_RachBungBinh_NguyenThong_1.h264",
        "file:///home/nvidia/Videos/trimmed_videos/trimmed_TranHungDao_NguyenVanCu.h264",
        "file:///home/nvidia/Videos/trimmed_videos/trimmed_TranKhacChan_TranQuangKhai.h264"
    ]
    sys.exit(main(stream_paths))
