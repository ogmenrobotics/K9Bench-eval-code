import json
import os
from zai import ZaiClient
import time
from datasets import load_dataset
import argparse
import yaml


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Config file {config_path} not found")
        raise
    except yaml.YAMLError as e:
        print(f"❌ Error parsing config file: {e}")
        raise


# ====================== MAIN ======================
def main():
    parser = argparse.ArgumentParser(description="Process QAs from the Hugging Face dataset.")
    parser.add_argument('--config', type=str, default="config.yaml", help='Path to config YAML file')
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index to process (inclusive).")
    parser.add_argument("--end_idx", type=int, default=None, help="Ending index to process (exclusive).")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to process.")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialize ZAI client
    client = ZaiClient(api_key=config['api_key'])

    # Override split from config if not provided in command line
    split = args.split or config.get('default_split', 'test')

    # Define folders and files from config
    HIGH_LEVEL_FOLDER = config['high_level_folder']
    OUTPUT_JSON = config['output_json']
    MODEL_NAME = config['model_name']
    PROMPT_FILE = config['prompt_file']          

    print(f"Loading dataset split: {split}")

    # Load dataset
    dataset = load_dataset(config['dataset_name'])
    video_qas = dataset[split]
    print(f"Processing split '{split}' with {len(video_qas)} examples")

    # Create proper slice
    if args.end_idx is None:
        end_idx = len(video_qas)
    else:
        end_idx = min(args.end_idx, len(video_qas))

    print(f"Processing indices {args.start_idx} to {end_idx}")

    # Load existing results
    processed_keys = set()
    paraphrased_results = []

    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                paraphrased_results = json.load(f)
            processed_keys = {
                (item["scene_name"], item["question_category"], item["paraphrased"]["correct_answer_key"])
                for item in paraphrased_results
                if "paraphrased" in item and "correct_answer_key" in item["paraphrased"]
            }
            print(f"Loaded {len(paraphrased_results)} existing entries.")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}. Starting fresh.")

    # Load prompt from configurable path
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            user_prompt_template = f.read()
        print(f"✅ Prompt loaded from {PROMPT_FILE}")
    except FileNotFoundError:
        print(f"❌ Prompt file not found: {PROMPT_FILE}")
        return
    except Exception as e:
        print(f"❌ Error reading prompt file: {e}")
        return

    # Process each QA
    for absolute_idx in range(args.start_idx, end_idx):
        try:
            qa = video_qas[absolute_idx]

            scene_name = qa["scene_name"]
            question_category = qa["question_category"]
            question = qa["question"]
            correct_answer = qa["ground_truth"]
            options = qa["options"]
            orig_idx = qa["idx"]

            # Skip already processed
            unique_key = (scene_name, question_category, correct_answer)
            if unique_key in processed_keys:
                print(f"Skipping already processed {scene_name} - {question_category}")
                continue

            print(f"Processing {absolute_idx}: {scene_name} - {question_category}")

            # Load high-level info
            high_level_file = os.path.join(HIGH_LEVEL_FOLDER, f"{scene_name}_high_level_info.json")
            if not os.path.exists(high_level_file):
                print(f"Warning: High-level info not found for {scene_name}, skipping.")
                continue

            with open(high_level_file, 'r', encoding='utf-8') as f:
                high_level_info = json.load(f)

            # Prepare data for prompt
            qa_json = json.dumps({
                "question": question,
                "Options": options,
                "correct_answer": correct_answer
            }, indent=2)

            high_level_json_str = json.dumps(high_level_info, indent=2)

            user_content = user_prompt_template.format(
                high_level_json=high_level_json_str,
                qa_json=qa_json,
                correct_answer_key=correct_answer
            )

            # API Call with retry
            success = False
            retries = 0
            while not success and retries < config.get('max_retries', 3):
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are a helpful AI assistant."},
                            {"role": "user", "content": user_content}
                        ]
                    )
                    success = True
                except Exception as e:
                    print(f"API error (retry {retries+1}): {e}")
                    retries += 1
                    if retries < config.get('max_retries', 3):
                        time.sleep(5)

            if not success:
                print(f"Failed after retries for {scene_name}")
                continue

            # Parse response
            response_content = response.choices[0].message.content.strip()
            if response_content.startswith("```json"):
                response_content = response_content[7:-3].strip()
            elif response_content.startswith("```"):
                response_content = response_content[3:-3].strip()

            paraphrased_json = json.loads(response_content)

            new_entry = {
                "idx": orig_idx,
                "processing_idx": absolute_idx,
                "scene_name": scene_name,
                "question_category": question_category,
                "paraphrased": paraphrased_json
            }

            paraphrased_results.append(new_entry)
            processed_keys.add(unique_key)

            # Save immediately
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(paraphrased_results, f, indent=4)

            print(f"✓ Processed & saved {scene_name} - {question_category}")
            time.sleep(config.get('sleep_between_requests', 1))

        except Exception as e:
            print(f"Error processing index {absolute_idx}: {e}")
            continue

    print(f"\n✅ Processing complete! Results saved to {OUTPUT_JSON}")
    print(f"Total entries: {len(paraphrased_results)}")


if __name__ == "__main__":
    main()
