import logging
import os
import cv2
import json
import shutil
import subprocess
import numpy as np
import google.generativeai as genai
import yaml
import time
# Load configuration from video_collection_pipeline.yaml
try:
    with open("../../configs/video_collection_pipeline.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
except FileNotFoundError:
    logging.error("video_collection_pipeline.yaml file not found in the working directory. Please create it with the required configuration.")
    raise
except yaml.YAMLError as e:
    logging.error(f"Error parsing video_collection_pipeline.yaml: {e}")
    raise
# Configure logging
logging.basicConfig(
    level=getattr(logging, config["logging"]["level"]),
    format=config["logging"]["format"]
)
# Directory and file paths from config
DOWNLOAD_DIR = config["download"]["directory"]
SUBSAMPLED_DIR = config["filtering"]["subsampled_directory"]
FINAL_DIR = config["filtering"]["final_directory"]
REJECT_LOG = config["filtering"]["reject_log_file"]
VIDEO_METADATA_FILE = os.path.join(config["output"]["directory"], config["output"]["video_details_file"])
FILTERED_IDS_FILE = os.path.join(config["output"]["directory"], config["filtering"]["filtered_ids_file"])
SKIPPED_LOG_FILE = config["filtering"]["skipped_log_file"]
GEMINI_API_KEY = config["filtering"]["gemini_api_key"]
GEMINI_MODEL = config["filtering"]["gemini_model"]
MAX_FRAMES = config["filtering"]["max_frames"]
KEEP_ORIGINAL = config["filtering"]["keep_original"]
# Create directories
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(SUBSAMPLED_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)
def load_video_metadata(video_id):
    if not os.path.exists(VIDEO_METADATA_FILE):
        logging.error(f"Video metadata file {VIDEO_METADATA_FILE} does not exist.")
        return "", ""
    with open(VIDEO_METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    video_info = metadata.get(video_id, {})
    return video_info.get("title", ""), video_info.get("description", "")
def transcode_if_av1(input_path):
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", input_path
        ], capture_output=True, text=True, check=True)
        codec = result.stdout.strip()
        if codec == "av1":
            logging.info(f"AV1 codec detected in {input_path}. Transcoding to H.264...")
            temp_path = input_path.replace(".mp4", "_h264.mp4")
            subprocess.run([
                "ffmpeg", "-i", input_path, "-c:v", "libx264", "-c:a", "copy", temp_path,
                "-y", "-hide_banner", "-loglevel", "error"
            ], check=True)
            os.replace(temp_path, input_path)
            logging.info(f"Transcoded to H.264: {input_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to detect or transcode codec for {input_path}: {e}")
def extract_frames(input_path, frames_dir, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        logging.error(f"Unable to read frames from: {input_path}")
        return 0
    os.makedirs(frames_dir, exist_ok=True)
    indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=int)
    idx_set = set(indices)
    count = 0
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num in idx_set:
            frame_path = os.path.join(frames_dir, f"frame_{count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            count += 1
        frame_num += 1
    cap.release()
    logging.info(f"Extracted {count} frames to {frames_dir}")
    return count
def make_1fps_video_from_frames(frames_dir, temp_video_path, frame_count):
    """Create a 1FPS video from extracted frames (no audio needed for Gemini)."""
    frame_pattern = os.path.join(frames_dir, "frame_%04d.jpg")
    result = subprocess.run([
        "ffmpeg", "-framerate", "1", "-i", frame_pattern, "-frames:v", str(frame_count),
        "-c:v", "libx264", "-an", "-pix_fmt", "yuv420p", temp_video_path,
        "-y", "-hide_banner", "-loglevel", "error"
    ], capture_output=True)
    if result.returncode != 0:
        logging.error(f"Failed to create 1FPS video from {frame_count} frames in {frames_dir}: {result.stderr.decode()}")
        return False
    logging.info(f"Created temp 1FPS video ({frame_count} frames) at {temp_video_path}")
    return True
def check_gemini(frames_dir, title, description, video_id, gemini_api_key, frame_count):
    temp_video_path = os.path.join(frames_dir, f"{video_id}_temp_1fps.mp4")
    try:
        # Step 1: Create 1FPS video from frames
        if not make_1fps_video_from_frames(frames_dir, temp_video_path, frame_count):
            return "Video Creation Failed"
        
        # Step 2: Upload the single video to Gemini
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        uploaded = genai.upload_file(path=temp_video_path, display_name=f"dog_video_frames_{video_id}")
        while uploaded.state.name == "PROCESSING":
            time.sleep(1)  # Poll every second
            uploaded = genai.get_file(uploaded.name)
        if uploaded.state.name == "FAILED":
            logging.error(f"Gemini upload failed for temp video of {video_id}")
            return "Upload Failed"
        
        # Step 3: Analyze with prompt
        prompt = (
            "Based on this 1FPS video created from evenly subsampled frames of the YouTube video, determine if the video is related to dogs, contains dogs, "
            "is not a compilation of multiple videos, and contains no sexual content.\n"
            "Strictly follow the criteria below:\n"
            "- The video must contain a real dog.\n"
            "- The video must not be artificially generated.\n"
            "- The video must not be a compilation.\n"
            "- The video must not include any inappropriate or sexual content.\n"
            "- The dog should be active and engaged in meaningful behavior (not stationary).\n"
            "- Human actions must be appropriate.\n"
            "\nRespond with 'YES' if the video fully meets the criteria. Otherwise, respond with 'NO'.\n"
            "If 'NO', give the exact reason strictly from the following list:\n"
            "- No Dog Present\n"
            "- Artificially Generated\n"
            "- Not Appropriate\n"
            "- Inappropriate actions by humans\n"
            "- Stationary Dog\n"
            f"\nTitle: {title}\nDescription: {description}"
        )
        response = model.generate_content([uploaded, prompt])
        genai.delete_file(uploaded.name)
        
        # Cleanup: Delete temp video
        os.remove(temp_video_path)
        
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini processing failed for video {video_id}: {e}")
        # Cleanup on error
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        try:
            if 'uploaded' in locals():
                genai.delete_file(uploaded.name)
        except:
            pass
        return f"Gemini Error: {str(e)}"
def log_rejection(video_id, reason):
    log = {}
    if os.path.exists(REJECT_LOG):
        with open(REJECT_LOG, 'r') as f:
            log = json.load(f)
    log[video_id] = reason
    with open(REJECT_LOG, 'w') as f:
        json.dump(log, f, indent=2)
    logging.info(f"Logged rejection for video {video_id}: {reason}")
def log_filtered(video_id):
    with open(FILTERED_IDS_FILE, 'a') as f:
        f.write(video_id + "\n")
    logging.info(f"Logged filtered video ID: {video_id}")
def log_skipped(video_id, reason="Unreadable or corrupted video"):
    with open(SKIPPED_LOG_FILE, 'a') as f:
        f.write(json.dumps({video_id: reason}) + "\n")
    logging.info(f"Logged skipped video {video_id}: {reason}")
def process_pipeline(video_id):
    video_path = os.path.join(DOWNLOAD_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        logging.warning(f"Video file not found: {video_path}")
        log_skipped(video_id, "Video file not found")
        return
    transcode_if_av1(video_path)
    title, description = load_video_metadata(video_id)
    frames_dir = os.path.join(SUBSAMPLED_DIR, video_id)
    frame_count = extract_frames(video_path, frames_dir)
    if frame_count == 0:
        logging.warning(f"Skipping {video_id} due to frame extraction failure.")
        log_skipped(video_id)
        shutil.rmtree(frames_dir, ignore_errors=True)
        os.remove(video_path)
        return
    logging.info(f"Starting Gemini analysis for video {video_id}")
    result = check_gemini(frames_dir, title, description, video_id, GEMINI_API_KEY, frame_count)
    shutil.rmtree(frames_dir, ignore_errors=True)
    if result.startswith("YES"):
        final_output = os.path.join(FINAL_DIR, f"{video_id}.mp4")
        shutil.copy2(video_path, final_output)
        log_filtered(video_id)
        os.remove(video_path)  # Always delete original after copying to final
    else:
        logging.info(f"Rejected video {video_id}: {result}")
        log_rejection(video_id, result)
        os.remove(video_path)  # Always delete original for rejected videos
# def main():
# video_ids_file = os.path.join(config["output"]["directory"], config["output"]["video_ids_file"])
# if not os.path.exists(video_ids_file):
# logging.error(f"Video IDs file {video_ids_file} does not exist. Please run youtube_agent.py first.")
# return
# with open(video_ids_file) as f:
# video_ids = [line.strip() for line in f if line.strip()]
# total_videos = len(video_ids)
# for idx, video_id in enumerate(video_ids, 1):
# logging.info(f"Processing video {idx}/{total_videos}: {video_id}")
# process_pipeline(video_id)
# if __name__ == "__main__":
# main()
