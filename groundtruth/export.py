import os
import subprocess

input_videos = ["/home/nvidia/Videos/DongKhoi_MacThiBuoi.mp4", "/home/nvidia/Videos/RachBungBinh_NguyenThong_1.mp4", 
                "/home/nvidia/Videos/TranHungDao_NguyenVanCu.mp4", "/home/nvidia/Videos/TranKhacChan_TranQuangKhai.mp4"]
output_folder = "groundtruth_input"

os.makedirs(output_folder, exist_ok=True)

def get_video_duration(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "format=duration", 
         "-of", "default=noprint_wrappers=1:nokey=1", video_path], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )
    return float(result.stdout.strip())

def extract_first_200_frames(input_file):
    """Extract the first 200 frames from the last 5 minutes of the input video."""
    duration = get_video_duration(input_file)
    last_5_min_start = max(0, duration - 300)

    output_file = os.path.join(output_folder, f"first_200_frames_{os.path.basename(input_file)}")
    ffmpeg_command = [
        "ffmpeg", "-ss", str(last_5_min_start), "-i", input_file,
        "-vf", "select=lt(n\,200)", "-vsync", "vfr", "-c:v", "libx264", "-preset", "fast", 
        "-crf", "23", output_file, "-y"
    ]
    
    subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Extracted first 200 frames saved: {output_file}")

for video in input_videos:
    if os.path.exists(video):
        extract_first_200_frames(video)
    else:
        print(f"File not found: {video}")

print("Processing completed!")
