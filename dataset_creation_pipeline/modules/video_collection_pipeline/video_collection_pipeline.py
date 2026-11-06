import logging
import os
import yaml
import argparse
from youtube_agent import YouTubeAgent
from youtube_video_downloader import YouTubeVideoDownloader
from video_filtering import process_pipeline
# Parse command-line arguments
parser = argparse.ArgumentParser(description="YouTube video processing pipeline")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--all", action="store_true", help="Run all pipeline steps")
group.add_argument("--collect_video", action="store_true", help="Run only video collection step")
group.add_argument("--download_video", action="store_true", help="Run only donwload video step")
group.add_argument("--video_filter", action="store_true", help="Run only video filtering step")
args = parser.parse_args()
# Load configuration from video_collection_pipeline.yaml
try:
    with open("../../configs/video_collection_pipeline.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
except FileNotFoundError:
    logging.error("video_collection_pipeline.yaml not found in ../../configs. Please create it with the required configuration.")
    raise
except yaml.YAMLError as e:
    logging.error(f"Error parsing video_collection_pipeline.yaml: {e}")
    raise
# Configure logging
logging.basicConfig(
    level=getattr(logging, config["logging"]["level"]),
    format=config["logging"]["format"]
)
googleapiclient_logger = logging.getLogger('googleapiclient')
googleapiclient_logger.setLevel(getattr(logging, config["logging"]["googleapiclient"]["level"]))
googleapiclient_logger.handlers = config["logging"]["googleapiclient"]["handlers"]
# Main workflow
def main():
    logging.info("Starting YouTube video processing pipeline...")
    if args.all or args.collect_video:
        # Step 1: Collect video IDs and metadata
        logging.info("Step 1: Collecting video IDs and metadata")
        youtube_agent = YouTubeAgent()
        video_details = youtube_agent.process_videos()
        if not video_details:
            logging.error("No videos collected. Exiting.")
            return
        video_ids = list(video_details.keys())
        logging.info(f"Collected {len(video_ids)} video IDs")
    if args.all or args.download_video:
        # Step 2: Download videos
        logging.info("Step 2: Downloading videos")
        downloader = YouTubeVideoDownloader()
        downloader.download_all_videos()
       
    if args.all or args.video_filter:
        # Step 3: Filter videos
        logging.info("Step 3: Filtering videos")
        video_ids_file = os.path.join(config["output"]["directory"], config["output"]["video_ids_file"])
        if not os.path.exists(video_ids_file):
            logging.error(f"Video IDs file {video_ids_file} does not exist. Please run youtube_agent.py first.")
            return
        with open(video_ids_file) as f:
            video_ids = [line.strip() for line in f if line.strip()]
        total_videos = len(video_ids)
        for idx, video_id in enumerate(video_ids, 1):
            logging.info(f"Processing video {idx}/{total_videos}: {video_id}")
            process_pipeline(video_id)
    logging.info("Pipeline completed.")
if __name__ == "__main__":
    main()
