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
    log_file = f"mcq_gen_{os.path.basename(config_path).replace('.yaml', '')}.log"
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
            self.qa_dir = config.get('qa_dir', 'qa')
            self.narration_dir = config.get('narration_dir', 'narrations')
            self.output_dir = config.get('output_dir', 'outputs')
            self.failed_log = config.get('failed_log', 'failed_videos.log')
            self.thought_dir = config.get('thought_dir', 'thoughts')
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

class GeminiMCQAgent:
    """Handles interaction with Gemini API for MCQ wrong answer generation."""
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

    def load_input_files(self, video_id, qa_dir, narration_dir):
        """Load QA and narration JSON files for a given video ID."""
        qa_path = os.path.join(qa_dir, f"{video_id}_qa.json")
        narration_path = os.path.join(narration_dir, f"{video_id}_narrations.json")
        try:
            with open(qa_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            logging.info(f"Loaded QA file: {qa_path}")
            print(f"📜 Loaded QA file: {qa_path}")
        except FileNotFoundError:
            logging.error(f"QA file {qa_path} not found")
            print(f"❌ QA file {qa_path} not found")
            return None, None
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing QA file {qa_path}: {e}")
            print(f"❌ Error parsing QA file {qa_path}: {e}")
            return None, None

        try:
            with open(narration_path, 'r', encoding='utf-8') as f:
                narration_data = json.load(f)
            logging.info(f"Loaded narration file: {narration_path}")
            print(f"📜 Loaded narration file: {narration_path}")
        except FileNotFoundError:
            logging.error(f"Narration file {narration_path} not found")
            print(f"❌ Narration file {narration_path} not found")
            return None, None
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing narration file {narration_path}: {e}")
            print(f"❌ Error parsing narration file {narration_path}: {e}")
            return None, None

        return qa_data, narration_data

    def generate_mcq(self, qa_data, narration_data, prompt_text, retries, video_id, output_dir, thought_dir):
        """Generate MCQ wrong answers based on QA and narration data."""
        if not self.client:
            logging.error("Client not initialized")
            print("❌ Client not initialized")
            return None, 0, False

        # Combine QA and narration with the prompt
        full_prompt = (
            f"{prompt_text.strip()}\n\n"
            f"## Question and Correct Answer Generated ##\n{json.dumps(qa_data, ensure_ascii=False, indent=2)}\n\n"
            f"## Video Narrations ##\n{json.dumps(narration_data, ensure_ascii=False, indent=2)}\n\n"
            f"[RequestID: {time.time()}-{random.randint(1000, 9999)}]"
        )

        for attempt in range(retries):
            try:
                logging.info(f"Generating MCQ (attempt {attempt + 1}/{retries})")
                print(f"🔄 Generating MCQ (attempt {attempt + 1}/{retries})")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=types.Content(
                        parts=[types.Part(text=full_prompt)]
                    ),
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            include_thoughts=True
                        )
                    )
                )

                # Extract thought summaries and save to a separate file
                thought_summaries = []
                for part in response.candidates[0].content.parts:
                    if part.thought and part.text:
                        thought_summaries.append(part.text)
                
                # Save thought summaries to {video_id}_thought_summaries.txt in thought_dir
                if thought_summaries:
                    os.makedirs(thought_dir, exist_ok=True)
                    thought_summary_path = os.path.join(thought_dir, f"{video_id}_thought_summaries.txt")
                    try:
                        with open(thought_summary_path, "w", encoding="utf-8") as f:
                            f.write("\n\n".join(thought_summaries))
                        logging.info(f"Saved thought summaries: {thought_summary_path}")
                        print(f"✅ Saved thought summaries: {thought_summary_path}")
                    except Exception as e:
                        logging.error(f"Failed to save thought summaries for {video_id}: {e}")
                        print(f"❌ Failed to save thought summaries for {video_id}: {e}")

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
                        # Validate each item and ensure exact match with qa_data
                        if len(parsed_output) != len(qa_data):
                            raise ValueError(f"Output contains {len(parsed_output)} questions, but input qa_data has {len(qa_data)} questions")
                        # Validate each item
                        for item in parsed_output:
                            if not all(key in item for key in ["question_category", "question", "correct_answer", "wrong_answers"]):
                                raise ValueError("Invalid MCQ item structure")
                            if not isinstance(item["wrong_answers"], list) or not (len(item["wrong_answers"]) == 4):
                                raise ValueError("Wrong answers must be a list of 4 items")
                            if item["question_category"] not in ["TYPE-I", "TYPE-II", "TYPE-III"]:
                                raise ValueError(f"Invalid question_category: {item['question_category']}")
                            # Ensure question, correct_answer, and question_category match input QA
                            matching_qa = next((qa for qa in qa_data if qa["question"] == item["question"]), None)
                            if not matching_qa or matching_qa["answer"] != item["correct_answer"] or matching_qa["question_category"] != item["question_category"]:
                                raise ValueError("Question, correct_answer, or question_category does not match input QA")
                        # Ensure all questions from qa_data are present in parsed_output
                        input_questions = {qa["question"] for qa in qa_data}
                        output_questions = {item["question"] for item in parsed_output}
                        if input_questions != output_questions:
                            raise ValueError("Output questions do not exactly match input QA questions")
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
                        parsed_output = self.create_fallback_output(qa_data)

                total_tokens = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0
                return parsed_output, total_tokens, False

            except Exception as e:
                logging.error(f"MCQ generation failed (attempt {attempt + 1}): {e}")
                print(f"❌ MCQ generation failed (attempt {attempt + 1}): {e}")
                if "429" in str(e) or "Quota" in str(e):
                    return None, 0, True
                if attempt < retries - 1:
                    logging.info("Retrying after 5 seconds...")
                    print("🔄 Retrying after 5 seconds...")
                    time.sleep(5)
                else:
                    logging.error("All retries failed")
                    print("❌ All retries failed")
                    return self.create_fallback_output(qa_data), 0, False

    def create_fallback_output(self, qa_data):
        """Create a fallback list structure using original QA data if parsing fails."""
        return [
            {
                "question_category": item["question_category"],
                "question": item["question"],
                "correct_answer": item["answer"],
                "wrong_answers": []
            } for item in qa_data
        ]

def save_mcq(video_id, mcq_data, output_dir, failed_log):
    """Save MCQ data to a JSON file, log failures separately."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_id}_mcq.json")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mcq_data, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved MCQ JSON: {output_path}")
        print(f"✅ Saved MCQ JSON: {output_path}")
    except Exception as e:
        error_msg = f"Failed to save MCQ JSON for {video_id}: {str(e)}"
        logging.error(error_msg)
        print(f"❌ Failed to save MCQ JSON for {video_id}: {str(e)}")
        with open(failed_log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")

def main():
    """Main function to process video IDs and generate MCQ wrong answers."""
    parser = argparse.ArgumentParser(description="Generate MCQ wrong answers using a config file.")
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
    agent = GeminiMCQAgent(api_keys[0], config.model_name)
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

    # Process each video ID
    for video_id in video_ids:
        logging.info(f"Processing video ID: {video_id}")
        print(f"📜 Processing video ID: {video_id}...")

        # Load QA and narration files
        qa_data, narration_data = agent.load_input_files(video_id, config.qa_dir, config.narration_dir)
        if qa_data is None or narration_data is None:
            error_msg = f"Skipping video ID {video_id} due to missing or invalid input files"
            logging.error(error_msg)
            print(f"❌ {error_msg}")
            with open(config.failed_log, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}\n")
            continue

        while True:
            mcq_data, total_tokens, quota_exhausted = agent.generate_mcq(
                qa_data, narration_data, prompt_text, config.max_retries, video_id, config.output_dir, config.thought_dir
            )

            if mcq_data is not None:
                save_mcq(video_id, mcq_data, config.output_dir, config.failed_log)
                break
            else:
                error_msg = f"No MCQ data generated for {video_id}"
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
                agent = GeminiMCQAgent(api_keys[key_index], config.model_name)
                if not agent.client:
                    logging.error("Failed to initialize new API key. Exiting.")
                    print("❌ Failed to initialize new API key. Exiting.")
                    return
            else:
                logging.error(f"Skipping video ID {video_id} after retries")
                print(f"❌ Skipping video ID {video_id} after retries")
                break

        # Calculate rate limit delay
        rpm_delay = 60 / config.rate_limit_rpm  # Seconds per request
        tpm_delay = total_tokens / (config.rate_limit_tpm / 60)  # Seconds based on tokens
        delay = max(rpm_delay, tpm_delay, 4)  # Minimum 4 seconds between videos
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
