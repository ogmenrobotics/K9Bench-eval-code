import os
from google import genai
from google.genai import types
import json
import time
import random
import logging
import yaml
import re
import argparse

def setup_logging(config_path):
    """Configure logging with a unique log file per config."""
    log_file = f"video_qa_{os.path.basename(config_path).replace('.yaml', '')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

class Config:
    """Load and store configuration from config.yaml."""
    def __init__(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.api_keys = config.get('api_keys', [])
            self.model_name = config.get('model_name', 'gemini-2.5-flash')
            self.video_ids_file = config.get('video_ids_file', 'video_ids.txt')
            self.prompt_file = config.get('prompt_file', 'prompt.txt')
            self.output_dir = config.get('output_dir', 'outputs')
            self.failed_log = config.get('failed_log', 'failed_videos.log')
            self.rate_limit_rpm = config.get('rate_limit_rpm', 15)
            self.rate_limit_tpm = config.get('rate_limit_tpm', 1000000)
            self.upload_timeout = config.get('upload_timeout', 300)
            self.max_retries = config.get('max_retries', 3)
            logging.info(f"Configuration loaded from {config_path}")
            print(f"✅ Configuration loaded from {config_path}")
        except FileNotFoundError:
            logging.error(f"Config file {config_path} not found")
            print(f"❌ Config file {config_path} not found")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing config file {config_path}: {e}")
            print(f"❌ Error parsing config file {config_path}: {e}")
            raise

class GeminiQAAgent:
    """Handles interaction with Gemini API for video QA generation."""
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        self.configure_client()

    def configure_client(self):
        """Initialize the Gemini client with the provided API key."""
        try:
            self.client = genai.Client(api_key=self.api_key)
            logging.info(f"Initialized client for model {self.model_name} with key: {self.api_key[:5]}...")
            print(f"✅ Initialized client for model {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize client for model {self.model_name}: {e}")
            print(f"❌ Failed to initialize client for model {self.model_name}: {e}")
            self.client = None

    def upload_video(self, video_id, retries):
        """Generate YouTube URL and prepare video for processing."""
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        logging.info(f"Preparing video URL: {youtube_url}")
        print(f"📺 Preparing video URL: {youtube_url}")
        return youtube_url, False

    def generate_qa(self, youtube_url, prompt_text, retries):
        """Generate QA pairs for a YouTube video."""
        if not self.client:
            logging.error("Client not initialized")
            print("❌ Client not initialized")
            return None, 0, False

        nonce = f"\n\n[RequestID: {time.time()}-{random.randint(1000, 9999)}]"
        full_prompt = prompt_text.strip() + nonce

        for attempt in range(retries):
            try:
                logging.info(f"Generating QA (attempt {attempt + 1}/{retries}) for URL: {youtube_url}")
                print(f"🔄 Generating QA (attempt {attempt + 1}/{retries}) for URL: {youtube_url}")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=types.Content(
                        parts=[
                            types.Part(file_data=types.FileData(file_uri=youtube_url)),
                            types.Part(text=full_prompt),
                        ]
                    ),
                )

                # Extract text from response
                description = ""
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text'):
                            description += part.text

                # Try to parse as JSON list
                parsed_output = None
                if description:
                    try:
                        json_match = re.search(r"```json\s*(.*?)\s*```", description, re.DOTALL)
                        json_text = json_match.group(1).strip() if json_match else description.strip()
                        parsed_output = json.loads(json_text)
                        if not isinstance(parsed_output, list):
                            raise ValueError("Parsed output is not a list")
                        # Validate that each item in the list has required keys
                        for item in parsed_output:
                            if not all(key in item for key in ["question_category", "question", "answer"]):
                                raise ValueError("Invalid QA item structure")
                    except (json.JSONDecodeError, ValueError) as e:
                        logging.warning(f"Invalid JSON or structure in attempt {attempt + 1}: {e}")
                        print(f"⚠️ Invalid JSON or structure in attempt {attempt + 1}: {e}")
                        if attempt < retries - 1:
                            logging.info("Retrying after 5 seconds due to invalid JSON...")
                            print("🔄 Retrying after 5 seconds due to invalid JSON...")
                            time.sleep(5)
                            continue
                        logging.error("All retries failed to produce valid JSON, falling back to template")
                        print("❌ All retries failed to produce valid JSON, falling back to template")
                        parsed_output = self.create_fallback_output(description)

                total_tokens = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                return parsed_output, total_tokens, False

            except Exception as e:
                logging.error(f"QA generation failed (attempt {attempt + 1}): {e}")
                print(f"❌ QA generation failed (attempt {attempt + 1}): {e}")
                if "429" in str(e) or "Quota" in str(e):
                    return None, 0, True
                if attempt < retries - 1:
                    logging.info("Retrying after 5 seconds...")
                    print("🔄 Retrying after 5 seconds...")
                    time.sleep(5)
                else:
                    logging.error(f"All {retries} retries failed for {youtube_url}")
                    print(f"❌ All {retries} retries failed for {youtube_url}")
                    return self.create_fallback_output(None), 0, False

    def create_fallback_output(self, description):
        """Create a fallback list structure if parsing fails."""
        return []  # Empty list for QA pairs

def save_qa(video_id, qa_data, output_dir, failed_log):
    """Save QA data to a JSON file, log failures separately."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_id}_qa.json")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved QA JSON: {output_path}")
        print(f"✅ Saved QA JSON: {output_path}")
    except Exception as e:
        error_msg = f"Failed to save QA JSON for {video_id}: {str(e)}"
        logging.error(error_msg)
        print(f"❌ Failed to save QA JSON for {video_id}: {str(e)}")
        with open(failed_log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")

def main():
    """Main function to process videos and generate QA pairs."""
    parser = argparse.ArgumentParser(description="Generate QA for videos using a config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
    args = parser.parse_args()

    setup_logging(args.config)
    config = Config(args.config)
    api_keys = config.api_keys
    if not api_keys:
        logging.error("No valid API keys provided. Exiting.")
        print("❌ No valid API keys provided. Exiting.")
        return

    # Initialize agent with first API key
    agent = GeminiQAAgent(api_keys[0], config.model_name)
    key_index = 0
    if not agent.client:
        logging.error("Client initialization failed with first key. Exiting.")
        print("❌ Client initialization failed with first key. Exiting.")
        return

    # Read video IDs
    try:
        with open(config.video_ids_file, "r", encoding="utf-8") as f:
            video_ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"Video IDs file {config.video_ids_file} not found")
        print(f"❌ Video IDs file {config.video_ids_file} not found")
        return

    # Read prompt
    try:
        with open(config.prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        logging.error(f"Prompt file {config.prompt_file} not found")
        print(f"❌ Prompt file {config.prompt_file} not found")
        return

    # Process each video
    for video_id in video_ids:
        logging.info(f"Processing video: {video_id}")
        print(f"📺 Processing video: {video_id}...")
        while True:
            youtube_url, quota_exhausted = agent.upload_video(video_id, config.max_retries)
            if youtube_url:
                break
            if quota_exhausted:
                key_index = (key_index + 1) % len(api_keys)
                if key_index == 0:
                    logging.error("All API keys exhausted. Exiting.")
                    print("❌ All API keys exhausted. Exiting.")
                    return
                logging.info(f"Quota exhausted. Switching to API key {key_index + 1}/{len(api_keys)}")
                print(f"🔄 Quota exhausted. Switching to API key {key_index + 1}/{len(api_keys)}")
                agent = GeminiQAAgent(api_keys[key_index], config.model_name)
                if not agent.client:
                    logging.error("Failed to initialize new API key. Exiting.")
                    print("❌ Failed to initialize new API key. Exiting.")
                    return
            else:
                logging.error(f"Skipping video {video_id} after retries")
                print(f"❌ Skipping video {video_id} after retries")
                break

        qa_data, total_tokens, quota_exhausted = agent.generate_qa(
            youtube_url, prompt_text, config.max_retries
        )

        if qa_data is not None:
            save_qa(video_id, qa_data, config.output_dir, config.failed_log)
        else:
            error_msg = f"No QA data generated for {video_id}"
            logging.error(error_msg)
            print(f"❌ {error_msg}")
            with open(config.failed_log, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")

        # Calculate rate limit delay
        rpm_delay = 60 / config.rate_limit_rpm  # Seconds per request
        tpm_delay = total_tokens / (config.rate_limit_tpm / 60)  # Seconds based on tokens
        delay = max(rpm_delay, tpm_delay, 4)  # Minimum 4 seconds between videos
        logging.info(f"Waiting {delay:.2f} seconds to respect rate limits...")
        print(f"⏳ Waiting {delay:.2f} seconds to respect rate limits...")
        time.sleep(delay)

        if quota_exhausted:
            key_index = (key_index + 1) % len(api_keys)
            if key_index == 0:
                logging.error("All API keys exhausted. Exiting.")
                print("❌ All API keys exhausted. Exiting.")
                return
            logging.info(f"Quota exhausted. Switching to API key {key_index + 1}/{len(api_keys)}")
            print(f"🔄 Quota exhausted. Switching to API key {key_index + 1}/{len(api_keys)}")
            agent = GeminiQAAgent(api_keys[key_index], config.model_name)
            if not agent.client:
                logging.error("Failed to initialize new API key. Exiting.")
                print("❌ Failed to initialize new API key. Exiting.")
                return

if __name__ == "__main__":
    main()
