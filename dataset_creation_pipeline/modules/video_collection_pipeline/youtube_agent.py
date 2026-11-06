import logging
import os
import json
import time
import yaml
from googleapiclient.discovery import build
# Load configuration from video_collection_pipeline.yaml
with open("../../configs/video_collection_pipeline.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)
# Configure logging
logging.basicConfig(
    level=getattr(logging, config["logging"]["level"]),
    format=config["logging"]["format"]
)
googleapiclient_logger = logging.getLogger('googleapiclient')
googleapiclient_logger.setLevel(getattr(logging, config["logging"]["googleapiclient"]["level"]))
googleapiclient_logger.handlers = config["logging"]["googleapiclient"]["handlers"]
# Load YouTube API keys from file
def load_youtube_api_keys(filename=config["youtube_api"]["key_file"]):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]
# Load video IDs from file
def load_video_ids(filename):
    if not os.path.exists(filename):
        logging.error(f"Video ID file {filename} does not exist.")
        return []
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]
# API Key Manager for switching keys
class APIKeyManager:
    def __init__(self, keys):
        self.keys = keys
        self.index = 0
    def get_key(self):
        return self.keys[self.index]
    def next_key(self):
        self.index = (self.index + 1) % len(self.keys)
        print(f"Switching to the next API key: Key #{self.index + 1} of {len(self.keys)}")
        return self.keys[self.index]
class YouTubeAgent:
    def __init__(self, youtube_key_file=config["youtube_api"]["key_file"]):
        self.youtube_keys = APIKeyManager(load_youtube_api_keys(youtube_key_file))
        print(f"Total YouTube API Keys: {len(self.youtube_keys.keys)}")
        self.youtube = self._init_youtube_client()
    def _init_youtube_client(self):
        return build("youtube", "v3", developerKey=self.youtube_keys.get_key())
    def execute_with_retries(self, method_name, *args, **kwargs):
        retries = len(self.youtube_keys.keys)
        delay = config["youtube_api"]["retry_delay"]
        attempt = 0
        while attempt < retries:
            current_key = self.youtube_keys.get_key()
            try:
                youtube_client = build("youtube", "v3", developerKey=current_key)
                if method_name == "search":
                    request = youtube_client.search().list(*args, **kwargs)
                elif method_name == "videos":
                    request = youtube_client.videos().list(*args, **kwargs)
                elif method_name == "channels":
                    request = youtube_client.channels().list(*args, **kwargs)
                elif method_name == "playlistItems":
                    request = youtube_client.playlistItems().list(*args, **kwargs)
                else:
                    raise ValueError(f"Unsupported method_name: {method_name}")
                response = request.execute()
                return response
            except Exception as e:
                error_msg = str(e)
                logging.warning(f"API error on attempt {attempt + 1}: {error_msg}")
                if "quotaExceeded" in error_msg or "403" in error_msg:
                    print(f"Quota exceeded for YouTube key #{self.youtube_keys.index + 1}. Switching key...")
                    self.youtube_keys.next_key()
                time.sleep(delay)
                delay = min(delay * 2, config["youtube_api"]["max_delay"])
                attempt += 1
        print(f"All {retries} YouTube API keys exhausted.")
        return None
    def get_video_ids_from_query(self, query, max_results=config["input"]["query_max_results"]):
        response = self.execute_with_retries("search", part="snippet", q=query, maxResults=max_results, type="video")
        if not response or "items" not in response:
            return []
        return [item["id"]["videoId"] for item in response["items"]]
    def get_all_videos_from_channel(self, channel_id, max_results=None):
        videos = []
        try:
            # Step 1: Get the uploads playlist ID
            response = self.execute_with_retries(
                "channels",
                part="contentDetails",
                id=channel_id
            )
            if not response or "items" not in response:
                logging.error(f"No channel data found for channel ID {channel_id}")
                return videos
            uploads_playlist_id = response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
            # Step 2: Paginate through all videos in the uploads playlist
            next_page_token = None
            total_results = 0
            while True:
                playlist_response = self.execute_with_retries(
                    "playlistItems",
                    part="snippet",
                    playlistId=uploads_playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                )
                if not playlist_response or "items" not in playlist_response:
                    logging.error(f"No playlist items found for playlist ID {uploads_playlist_id}")
                    break
                for item in playlist_response.get("items", []):
                    video_id = item["snippet"]["resourceId"]["videoId"]
                    videos.append(video_id)
                    total_results += 1
                logging.info(f"Fetched {len(playlist_response['items'])} videos from channel {channel_id}. Total: {total_results}")
                next_page_token = playlist_response.get("nextPageToken")
                if max_results and total_results >= max_results:
                    break
                if not next_page_token:
                    break
        except Exception as e:
            logging.error(f"Error fetching channel videos for channel ID {channel_id}: {e}")
        return videos[:max_results] if max_results else videos
    def get_video_details(self, video_id):
        video_ids = video_id if isinstance(video_id, list) else [video_id]
        response = self.execute_with_retries("videos", part="snippet,contentDetails", id=",".join(video_ids))
        if not response or "items" not in response or len(response["items"]) == 0:
            return {} if isinstance(video_id, list) else None
        result = {}
        for video in response["items"]:
            video_id = video["id"]
            snippet = video["snippet"]
            content_details = video["contentDetails"]
            result[video_id] = {
                "videoId": video_id,
                "title": snippet["title"],
                "description": snippet["description"],
                "channelId": snippet["channelId"],
                "publishedAt": snippet["publishedAt"],
                "duration": content_details["duration"]
            }
        return result if len(result) > 1 else result.get(video_id)
    def collect_video_details(self, initial_video_ids, max_results_per_video=config["processing"]["max_results_per_video"], max_videos_from_channel=config["processing"]["max_videos_from_channel"]):
        all_video_ids = set(initial_video_ids)
        video_sources = {} # Track source: 'initial', 'channel', or 'recommended'
        channel_video_map = {} # Map video ID to its source channel ID
        for idx, video_id in enumerate(initial_video_ids, 1):
            logging.info(f"Processing initial video {idx}/{len(initial_video_ids)}: {video_id}")
            video_details = self.get_video_details(video_id)
            if video_details:
                channel_id = video_details["channelId"]
                video_sources[video_id] = "initial"
                channel_video_map[video_id] = channel_id
                # Get all videos from the channel
                channel_video_ids = self.get_all_videos_from_channel(channel_id, max_results=max_videos_from_channel)
                for vid in channel_video_ids:
                    video_sources[vid] = "channel"
                    channel_video_map[vid] = channel_id
                all_video_ids.update(channel_video_ids)
                # Get recommended videos
                recommended_video_ids = self.get_video_ids_from_query(video_details["title"], max_results=max_results_per_video)
                for vid in recommended_video_ids:
                    video_sources[vid] = "recommended"
                    channel_video_map[vid] = channel_id
                all_video_ids.update(recommended_video_ids)
        all_video_details = {}
        video_id_batches = [list(all_video_ids)[i:i+50] for i in range(0, len(all_video_ids), 50)]
        for batch in video_id_batches:
            details = self.get_video_details(batch)
            if details:
                for vid, detail in details.items():
                    all_video_details[vid] = {
                        "details": detail,
                        "source": video_sources[vid],
                        "source_channel": channel_video_map[vid]
                    }
        return all_video_details
    def process_videos(self, input_source=config["input"]["source"], output_dir=config["output"]["directory"],
                       max_results_per_video=config["processing"]["max_results_per_video"],
                       max_videos_from_channel=config["processing"]["max_videos_from_channel"]):
        if input_source is None:
            raise ValueError("Input source must be provided.")
        if input_source.endswith('.txt'):
            initial_video_ids = load_video_ids(input_source)
        else:
            initial_video_ids = self.get_video_ids_from_query(input_source, max_results=config["input"]["query_max_results"])
        if not initial_video_ids:
            print("No video IDs found.")
            return {}
        # Collect all video details
        all_video_details = self.collect_video_details(initial_video_ids, max_results_per_video, max_videos_from_channel)
        os.makedirs(output_dir, exist_ok=True)
       
        # Save all video details
        with open(os.path.join(output_dir, config["output"]["video_details_file"]), "w") as f:
            json.dump({vid: info["details"] for vid, info in all_video_details.items()}, f, indent=4)
        with open(os.path.join(output_dir, config["output"]["video_ids_file"]), "w") as f:
            for vid in all_video_details.keys():
                f.write(f"{vid}\n")
        with open(os.path.join(output_dir, config["output"]["video_sources_file"]), "w") as f:
            json.dump({vid: {"source": info["source"], "source_channel": info["source_channel"]} for vid, info in all_video_details.items()}, f, indent=4)
        print(f"All video details saved to {output_dir}/{config['output']['video_details_file']}")
        print(f"Video sources saved to {output_dir}/{config['output']['video_sources_file']}")
        return all_video_details
# def main():
# youtube_agent = YouTubeAgent()
# video_details = youtube_agent.process_videos()
# print(f"Processed {len(video_details)} videos.")
# if __name__ == "__main__":
# main()
