import os
import google.generativeai as genai
import json
import time
import random
import logging
import yaml
import re
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_narration.log'),
        logging.StreamHandler()
    ]
)

class Config:
    """Load and store configuration from config.yaml."""
    def __init__(self, config_path="../../configs/narration_config.yaml"):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.api_keys = config.get('api_keys', [])
            self.model_name = config.get('model_name', 'gemini-2.5-flash')
            self.video_ids_file = config.get('video_ids_file', 'video_ids.txt')
            self.prompt_file = config.get('prompt_file', 'prompt.txt')
            self.output_dir = config.get('output_dir', 'outputs')
            self.failed_log = config.get('failed_log', 'failed_videos.log')
            self.rate_limit_rpm = config.get('rate_limit_rpm', 10)  # Requests per minute
            self.rate_limit_tpm = config.get('rate_limit_tpm', 250000)  # Tokens per minute
            self.upload_timeout = config.get('upload_timeout', 300)  # Seconds
            self.max_retries = config.get('max_retries', 3)
            logging.info("Configuration loaded successfully")
        except FileNotFoundError:
            logging.error(f"Config file {config_path} not found")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing config file: {e}")
            raise

class GeminiNarrationAgent:
    """Handles interaction with Gemini API for video narration generation."""
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.model = None
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        self.configure_model()

    def configure_model(self):
        """Initialize the Gemini model with the provided API key."""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=self.safety_settings
            )
            logging.info(f"Initialized model {self.model_name} with key: {self.api_key[:5]}...")
        except Exception as e:
            logging.error(f"Failed to initialize model {self.model_name}: {e}")
            self.model = None

    def upload_video(self, video_id, retries):
        """Generate YouTube URL and prepare video for processing."""
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        logging.info(f"Preparing video URL: {youtube_url}")
        return youtube_url, False

    def generate_narration(self, youtube_url, prompt_text, retries):
        """Generate narration for a YouTube video."""
        if not self.model:
            logging.error("Model not initialized")
            return None, 0, False

        nonce = f"\n\n[RequestID: {time.time()}-{random.randint(1000, 9999)}]"
        full_prompt = prompt_text.strip() + nonce

        for attempt in range(retries):
            try:
                logging.info(f"Generating narration (attempt {attempt + 1}/{retries}) for URL: {youtube_url}")
                response = self.model.generate_content(
                    [
                        {"file_data": {"file_uri": youtube_url}},
                        {"text": full_prompt}
                    ],
                    safety_settings=self.safety_settings
                )
                
                print(response)


                # Extract text from response
                description = ""
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'text'):
                            description += part.text

                # Try to parse as JSON
                parsed_output = None
                if description:
                    try:
                        json_match = re.search(r"```json\s*(.*?)\s*```", description, re.DOTALL)
                        json_text = json_match.group(1).strip() if json_match else description.strip()
                        parsed_output = json.loads(json_text)
                    except json.JSONDecodeError:
                        logging.warning("Model did not return valid JSON, falling back to template")
                        parsed_output = self.create_fallback_output(description)

                total_tokens = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                return parsed_output, total_tokens, False

            except Exception as e:
                logging.error(f"Narration generation failed (attempt {attempt + 1}): {e}")
                if "429" in str(e) or "Quota" in str(e):
                    return None, 0, True
                if attempt < retries - 1:
                    logging.info("Retrying after 5 seconds...")
                    time.sleep(5)
                else:
                    logging.error(f"All {retries} retries failed for {youtube_url}")
                    return self.create_fallback_output(None), 0, False

    def create_fallback_output(self, description):
        """Create a fallback JSON structure if parsing fails. Scenes are empty to allow for variable number."""
        return {
            "detailed_description": description.strip() if description else "Failed to generate description.",
            "scenes": {}  # Empty dict to handle variable scenes; no fixed number
        }

def save_narration(video_id, narration_data, output_dir, failed_log):
    """Save narration data to a JSON file, log failures separately."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_id}_narrations.json")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(narration_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved narration JSON: {output_path}")
    except Exception as e:
        error_msg = f"Failed to save narration JSON for {video_id}: {str(e)}"
        logging.error(error_msg)
        # Log to separate failed log
        with open(failed_log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")

def main():
    """Main function to process videos and generate narrations."""
    config = Config()
    api_keys = config.api_keys
    if not api_keys:
        logging.error("No valid API keys provided. Exiting.")
        return

    # Initialize agent with first API key
    agent = GeminiNarrationAgent(api_keys[0], config.model_name)
    key_index = 0
    if not agent.model:
        logging.error("Model initialization failed with first key. Exiting.")
        return

    # Read video IDs
    try:
        with open(config.video_ids_file, "r", encoding="utf-8") as f:
            video_ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"Video IDs file {config.video_ids_file} not found")
        return

    # Read prompt
    try:
        with open(config.prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()
    except FileNotFoundError:
        logging.error(f"Prompt file {config.prompt_file} not found")
        return

    # Process each video
    for video_id in video_ids:
        logging.info(f"Processing video: {video_id}")
        while True:
            youtube_url, quota_exhausted = agent.upload_video(video_id, config.max_retries)
            if youtube_url:
                break
            if quota_exhausted:
                key_index = (key_index + 1) % len(api_keys)
                if key_index == 0:
                    logging.error("All API keys exhausted. Exiting.")
                    return
                logging.info(f"Quota exhausted. Switching to API key {key_index + 1}/{len(api_keys)}")
                agent = GeminiNarrationAgent(api_keys[key_index], config.model_name)
                if not agent.model:
                    logging.error("Failed to initialize new API key. Exiting.")
                    return
            else:
                logging.error(f"Skipping video {video_id} after retries")
                break

        narration_data, total_tokens, quota_exhausted = agent.generate_narration(
            youtube_url, prompt_text, config.max_retries
        )

        if narration_data:
            save_narration(video_id, narration_data, config.output_dir, config.failed_log)
        else:
            error_msg = f"No narration data generated for {video_id}"
            logging.error(error_msg)
            with open(config.failed_log, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")

        # Calculate rate limit delay
        rpm_delay = 60 / config.rate_limit_rpm  # Seconds per request
        tpm_delay = total_tokens / (config.rate_limit_tpm / 60)  # Seconds based on tokens
        delay = max(rpm_delay, tpm_delay, 4)  # Minimum 4 seconds between videos
        logging.info(f"Waiting {delay:.2f} seconds to respect rate limits...")
        time.sleep(delay)

        if quota_exhausted:
            key_index = (key_index + 1) % len(api_keys)
            if key_index == 0:
                logging.error("All API keys exhausted. Exiting.")
                return
            logging.info(f"Quota exhausted. Switching to API key {key_index + 1}/{len(api_keys)}")
            agent = GeminiNarrationAgent(api_keys[key_index], config.model_name)
            if not agent.model:
                logging.error("Failed to initialize new API key. Exiting.")
                return

if __name__ == "__main__":
    main()
