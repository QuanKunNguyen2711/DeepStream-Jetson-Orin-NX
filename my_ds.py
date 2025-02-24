import sys
sys.path.append('../')
import os
import gi
import numpy as np
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst, GstRtspServer
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call

import pyds
import configparser

PGIE_CLASS_ID_MOTORBIKE = 3
PGIE_CLASS_ID_CAR = 2
PGIE_CLASS_ID_TRUCK = 5
PGIE_CLASS_ID_BUS = 1
PGIE_CLASS_ID_VAN = 6
PGIE_CLASS_ID_BICYCLE = 0
PGIE_CLASS_ID_PEDESTRIAN = 4

def osd_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Initialize object counter for your custom classes
        obj_counter = {
            PGIE_CLASS_ID_MOTORBIKE: 0,
            PGIE_CLASS_ID_CAR: 0,
            PGIE_CLASS_ID_TRUCK: 0,
            PGIE_CLASS_ID_BUS: 0,
            PGIE_CLASS_ID_VAN: 0,
            PGIE_CLASS_ID_BICYCLE: 0,
            PGIE_CLASS_ID_PEDESTRIAN: 0
        }

        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            # print("BEGIN----------------------------------DEEPSTREAM---------------------------------")
            # print(f"class={obj_meta.class_id}, left={obj_meta.rect_params.left}, "
            #     f"top={obj_meta.rect_params.top}, width={obj_meta.rect_params.width}, "
            #     f"height={obj_meta.rect_params.height}, conf={obj_meta.confidence}")
            
            # Increment the count for the detected class
            obj_counter[obj_meta.class_id] += 1

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

        # Acquire display meta for drawing text
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]

        # Set display text to show counts for your custom classes
        py_nvosd_text_params.display_text = (
            f"Frame Number={frame_number} "
            f"Number of Objects={num_rects} "
            f"Motorbike={obj_counter[PGIE_CLASS_ID_MOTORBIKE]} "
            f"Car={obj_counter[PGIE_CLASS_ID_CAR]} "
            f"Truck={obj_counter[PGIE_CLASS_ID_TRUCK]} "
            f"Bus={obj_counter[PGIE_CLASS_ID_BUS]} "
            f"Van={obj_counter[PGIE_CLASS_ID_VAN]} "
            f"Bicycle={obj_counter[PGIE_CLASS_ID_BICYCLE]} "
            f"Pedestrian={obj_counter[PGIE_CLASS_ID_PEDESTRIAN]}"
        )

        
        # Set text position, font, and color
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 12
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)  # White text
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)  # Black background

        # Print the display text
        print(pyds.get_string(py_nvosd_text_params.display_text))

        # Add display meta to the frame
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def main(args):
    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <media file or uri>\n" % args[0])
        sys.exit(1)

    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements
    pipeline = Gst.Pipeline()
    source = Gst.ElementFactory.make("filesrc", "file-source")
    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    # Create a caps filter
    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
    
    # Make the encoder
    if codec == "H264":
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        print("Creating H264 Encoder")
    elif codec == "H265":
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        print("Creating H265 Encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder")
    encoder.set_property('bitrate', bitrate)
    
    # sink = Gst.ElementFactory.make("fakesink", "fake-video-renderer")
    # if not sink:
    #     sys.stderr.write(" Unable to create fakesink \n")

    if is_aarch64():
        encoder.set_property('preset-level', 1)
        encoder.set_property('insert-sps-pps', 1)
        print("Creating nv3dsink \n")
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
        if not sink:
            sys.stderr.write(" Unable to create nv3dsink \n")
    else:
        print("Creating EGLSink \n")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        if not sink:
            sys.stderr.write(" Unable to create egl sink \n")
            
    # Make the payload-encode video into RTP packets
    if codec == "H264":
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        print("Creating H264 rtppay")
    elif codec == "H265":
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
        print("Creating H265 rtppay")
    if not rtppay:
        sys.stderr.write(" Unable to create rtppay")
    
    # Make the UDP sink
    updsink_port_num = 5400
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not sink:
        sys.stderr.write(" Unable to create udpsink")
    
    sink.set_property('host', '224.224.255.255')
    sink.set_property('port', updsink_port_num)
    sink.set_property('async', False)
    sink.set_property('sync', 1)
    
    print("Playing file %s " % args[1])
    source.set_property('location', args[1])
    if os.environ.get('USE_NEW_NVSTREAMMUX') != 'yes':
        streammux.set_property('width', 1920)
        streammux.set_property('height', 1080)
        streammux.set_property('batched-push-timeout', 4000000)

    streammux.set_property('batch-size', 1)
    pgie.set_property('config-file-path', "myds_pgie_config.txt")

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

    # Add elements to the pipeline
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd) # RTSP
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)

    # Link elements in the pipeline
    # file-source -> h264-parser -> nvh264-decoder ->
    # nvinfer -> nvvidconv -> nvosd -> nvvidconv_postosd -> 
    # caps -> encoder -> rtppay -> udpsink
    source.link(h264parser)
    h264parser.link(decoder)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of decoder \n")

    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv)
    nvvidconv.link(nvosd)
    # nvosd.link(sink)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(sink)

    # Create an event loop and feed gstreamer bus messages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    
    # Start streaming
    rtsp_port_num = 8554
    
    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)
    
    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch( "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )" % (updsink_port_num, codec))
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds", factory)
    
    print("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds ***\n\n" % rtsp_port_num)

    # Add probe to get metadata
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # Start playback
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # Cleanup
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    codec = "H264"
    bitrate = 4000000
    sys.exit(main(sys.argv))