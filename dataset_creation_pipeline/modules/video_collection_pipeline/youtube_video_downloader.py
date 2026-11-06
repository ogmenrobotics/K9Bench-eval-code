import logging
import os
import yaml
import yt_dlp

# Load configuration from config.yaml
try:
    with open("../../configs/video_collection_pipeline.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
except FileNotFoundError:
    logging.error("config.yaml file not found in the working directory. Please create it with the required configuration.")
    raise
except yaml.YAMLError as e:
    logging.error(f"Error parsing config.yaml: {e}")
    raise

# Configure logging
logging.basicConfig(
    level=getattr(logging, config["logging"]["level"]),
    format=config["logging"]["format"]
)

# Load video IDs from file
def load_video_ids(filename):
    if not os.path.exists(filename):
        logging.error(f"Video ID file {filename} does not exist.")
        return []
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]

class YouTubeVideoDownloader:
    def __init__(self, video_ids_file=os.path.join(config["output"]["directory"], config["output"]["video_ids_file"]), 
                 download_dir=config["download"]["directory"]):
        self.video_ids_file = video_ids_file
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    def download_video(self, video_id):
        video_path = os.path.join(self.download_dir, f"{video_id}.mp4")
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            "format": "bv*+ba/best[ext=mp4]/best",
            "outtmpl": video_path,
            "merge_output_format": "mp4",
            "quiet": False,
            "no_warnings": False,
            "ignoreerrors": True,  # Continue on download errors
            "force_overwrites": True,  # Overwrite existing files
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            if not os.path.exists(video_path):
                raise Exception("Download completed but output file not found.")
            logging.info(f"Successfully downloaded video {video_id} to {video_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to download video {video_id}: {str(e)}")
            return False

    def download_all_videos(self):
        video_ids = load_video_ids(self.video_ids_file)
        if not video_ids:
            logging.warning(f"No video IDs found in {self.video_ids_file}. Nothing to download.")
            return

        total_videos = len(video_ids)
        successful_downloads = 0
        for idx, video_id in enumerate(video_ids, 1):
            logging.info(f"Processing video {idx}/{total_videos}: {video_id}")
            if self.download_video(video_id):
                successful_downloads += 1
            else:
                logging.warning(f"Skipping video {video_id} due to download failure.")

        print(f"Downloaded {successful_downloads}/{total_videos} videos successfully.")

def main():
    downloader = YouTubeVideoDownloader()
    downloader.download_all_videos()

if __name__ == "__main__":
    main()
