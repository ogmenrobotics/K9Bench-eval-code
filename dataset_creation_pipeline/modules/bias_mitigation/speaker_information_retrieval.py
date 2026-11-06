import os
import google.generativeai as genai
import json
import time
import logging
import yaml
import re
import argparse

def setup_logging(config_path):
    """Configure logging with a unique log file per config."""
    log_file = f"dog_info_gen_{os.path.basename(config_path).replace('.yaml', '')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s - [PID: %(process)d]',
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
            self.model_name = config.get('model_name', 'gemini-1.5-flash')
            self.video_ids_file = config.get('video_ids_file', 'video_ids.txt')
            self.narration_dir = config.get('narration_dir', 'Narrations')
            self.output_dir = config.get('output_dir', 'outputs')
            self.failed_log = config.get('failed_log', 'failed_videos.log')
            self.prompt_file = config.get('prompt_file', 'prompt_dog_info.txt')  # ← Added
            self.rate_limit_rpm = config.get('rate_limit_rpm', 15)
            self.rate_limit_tpm = config.get('rate_limit_tpm', 1000000)
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


class GeminiDogInfoAgent:
    """Handles interaction with Gemini API for dog info extraction."""
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        self.configure_client()

    def configure_client(self):
        """Initialize the Gemini client with the provided API key."""
        try:
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
            logging.info(f"Initialized client for model {self.model_name} with key: {self.api_key[:5]}...")
            print(f"✅ Initialized client for model {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize client for model {self.model_name}: {e}")
            print(f"❌ Failed to initialize client for model {self.model_name}: {e}")
            self.client = None

    def load_narration_file(self, video_id, narration_dir):
        """Load narration JSON file for a given video ID and extract narration text."""
        narration_path = os.path.join(narration_dir, f"{video_id}_narrations.json")
        try:
            with open(narration_path, 'r', encoding='utf-8') as f:
                narration_data = json.load(f)
            logging.info(f"Loaded narration file: {narration_path}")
            print(f"📜 Loaded narration file: {narration_path}")

            narration_text = ""
            if 'detailed_description' in narration_data:
                narration_text += narration_data['detailed_description'] + "\n\n"
            if 'scenes' in narration_data and isinstance(narration_data['scenes'], dict):
                for scene_key, scene_data in narration_data['scenes'].items():
                    if 'narration' in scene_data:
                        narration_text += f"Scene {scene_key} ({scene_data.get('time_interval', 'unknown time')}):\n"
                        narration_text += scene_data['narration'] + "\n\n"

            if not narration_text.strip():
                logging.warning(f"No valid narration text found in {narration_path}")
                print(f"⚠️ No valid narration text found in {narration_path}")
                return None

            return narration_text
        except FileNotFoundError:
            logging.error(f"Narration file {narration_path} not found")
            print(f"❌ Narration file {narration_path} not found")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing narration file {narration_path}: {e}")
            print(f"❌ Error parsing narration file {narration_path}: {e}")
            return None

    def generate_dog_info(self, narration, retries, video_id, output_dir, prompt_template):
        """Generate structured dog info based on narration."""
        if not self.client:
            logging.error("Client not initialized")
            print("❌ Client not initialized")
            return None, 0, False

        full_prompt = prompt_template.format(narration=narration)

        for attempt in range(retries):
            try:
                logging.info(f"Generating dog info (attempt {attempt + 1}/{retries})")
                print(f"🔄 Generating dog info (attempt {attempt + 1}/{retries})")

                response = self.client.generate_content(full_prompt)

                # Extract text from response
                description = response.text if hasattr(response, 'text') else ""

                # Try to parse as JSON
                parsed_output = None
                if description:
                    try:
                        json_match = re.search(r"```json\s*(.*?)\s*```", description, re.DOTALL)
                        json_text = json_match.group(1).strip() if json_match else description.strip()
                        parsed_output = json.loads(json_text)

                        # Basic validation
                        if not isinstance(parsed_output, dict):
                            raise ValueError("Parsed output is not a dict")
                        if "number_of_dogs" not in parsed_output or "dogs" not in parsed_output:
                            raise ValueError("Invalid structure: missing number_of_dogs or dogs")
                        if not isinstance(parsed_output["dogs"], dict):
                            raise ValueError("Dogs must be a dict")
                        for dog_key, dog_data in parsed_output["dogs"].items():
                            if not all(key in dog_data for key in ["name", "breed", "age", "color"]):
                                raise ValueError(f"Invalid dog data for {dog_key}: missing required fields")
                            if not dog_data["color"]:
                                raise ValueError(f"Color for {dog_key} cannot be empty")
                            if dog_data["age"] not in ["puppy", "adult", "old", ""]:
                                raise ValueError(f"Invalid age for {dog_key}: {dog_data['age']}")
                    except (json.JSONDecodeError, ValueError) as e:
                        logging.warning(f"Invalid JSON or structure in attempt {attempt + 1}: {e}")
                        print(f"⚠️ Invalid JSON or structure in attempt {attempt + 1}: {e}")
                        if attempt < retries - 1:
                            time.sleep(5)
                            continue
                        logging.error("All retries failed to produce valid JSON, falling back to template")
                        print("❌ All retries failed to produce valid JSON, falling back to template")
                        parsed_output = self.create_fallback_output()

                # Handle token count
                total_tokens = 0
                try:
                    if hasattr(response, 'usage_metadata') and hasattr(response.usage_metadata, 'total_token_count'):
                        total_tokens = response.usage_metadata.total_token_count
                except Exception:
                    pass

                return parsed_output, total_tokens, False

            except Exception as e:
                logging.error(f"Dog info generation failed (attempt {attempt + 1}): {e}")
                print(f"❌ Dog info generation failed (attempt {attempt + 1}): {e}")
                if "429" in str(e) or "Quota" in str(e):
                    return None, 0, True
                if attempt < retries - 1:
                    time.sleep(5)
                else:
                    logging.error("All retries failed")
                    print("❌ All retries failed")
                    return self.create_fallback_output(), 0, False

    def create_fallback_output(self):
        """Create a fallback dict structure if parsing fails."""
        return {
            "number_of_dogs": 0,
            "dogs": {}
        }


def save_dog_info(video_id, dog_info_data, output_dir, failed_log):
    """Save dog info data to a JSON file, log failures separately."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_id}_high_level_info.json")
    try:
        if dog_info_data.get("number_of_dogs", 0) > 0 or dog_info_data.get("dogs", {}):
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(dog_info_data, f, ensure_ascii=False, indent=4)
            logging.info(f"Saved dog info JSON: {output_path}")
            print(f"✅ Saved dog info JSON: {output_path}")
        else:
            logging.warning(f"Skipping save for {video_id}: Empty or invalid dog info data")
            print(f"⚠️ Skipping save for {video_id}: Empty or invalid dog info data")
    except Exception as e:
        error_msg = f"Failed to save dog info JSON for {video_id}: {str(e)}"
        logging.error(error_msg)
        print(f"❌ Failed to save dog info JSON for {video_id}: {str(e)}")
        with open(failed_log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")


def main():
    """Main function to process video IDs and generate dog info."""
    parser = argparse.ArgumentParser(description="Generate dog info using a config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
    args = parser.parse_args()

    setup_logging(args.config)
    config = Config(args.config)

    # Load prompt from config
    try:
        with open(config.prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        print(f"✅ Prompt loaded from {config.prompt_file}")
    except FileNotFoundError:
        print(f"❌ Prompt file not found: {config.prompt_file}")
        return
    except Exception as e:
        print(f"❌ Error reading prompt file: {e}")
        return

    api_keys = config.api_keys
    if not api_keys:
        logging.error("No valid API keys provided. Exiting.")
        print("❌ No valid API keys provided. Exiting.")
        return

    agent = GeminiDogInfoAgent(api_keys[0], config.model_name)
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

    # Process each video ID
    for video_id in video_ids:
        logging.info(f"Processing video ID: {video_id}")
        print(f"📜 Processing video ID: {video_id}...")

        narration = agent.load_narration_file(video_id, config.narration_dir)
        if narration is None:
            error_msg = f"Skipping video ID {video_id} due to missing or invalid narration file"
            logging.error(error_msg)
            print(f"❌ {error_msg}")
            with open(config.failed_log, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
            continue

        while True:
            dog_info_data, total_tokens, quota_exhausted = agent.generate_dog_info(
                narration, config.max_retries, video_id, config.output_dir, prompt_template
            )

            if dog_info_data is not None:
                save_dog_info(video_id, dog_info_data, config.output_dir, config.failed_log)
                break
            else:
                error_msg = f"No dog info data generated for {video_id}"
                logging.error(error_msg)
                print(f"❌ {error_msg}")
                with open(config.failed_log, "a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")

            if quota_exhausted:
                key_index = (key_index + 1) % len(api_keys)
                if key_index == 0:
                    logging.error("All API keys exhausted. Exiting.")
                    print("❌ All API keys exhausted. Exiting.")
                    return
                logging.info(f"Quota exhausted. Switching to API key {key_index + 1}/{len(api_keys)}")
                print(f"🔄 Quota exhausted. Switching to API key {key_index + 1}/{len(api_keys)}")
                agent = GeminiDogInfoAgent(api_keys[key_index], config.model_name)
                if not agent.client:
                    logging.error("Failed to initialize new API key. Exiting.")
                    print("❌ Failed to initialize new API key. Exiting.")
                    return
            else:
                logging.error(f"Skipping video ID {video_id} after retries")
                print(f"❌ Skipping video ID {video_id} after retries")
                break

        # Rate limiting
        rpm_delay = 60 / config.rate_limit_rpm
        tpm_delay = total_tokens / (config.rate_limit_tpm / 60) if total_tokens > 0 else 0
        delay = max(rpm_delay, tpm_delay, 4)
        logging.info(f"Waiting {delay:.2f} seconds to respect rate limits...")
        print(f"⏳ Waiting {delay:.2f} seconds to respect rate limits...")
        time.sleep(delay)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Script interrupted by user (Ctrl+C). Exiting gracefully.")
        print("❌ Script interrupted by user (Ctrl+C). Exiting gracefully.")
    except SystemExit:
        logging.info("Script suspended by user (Ctrl+Z). To resume, use 'fg' in the terminal.")
        print("⏸️ Script suspended by user (Ctrl+Z). To resume, use 'fg' in the terminal.")
