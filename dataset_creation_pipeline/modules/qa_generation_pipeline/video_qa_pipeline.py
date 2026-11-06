import logging
import os
import yaml
import argparse
import subprocess

# Parse command line arguments
parser = argparse.ArgumentParser(description="Video QA Processing Pipeline")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--all", action="store_true", help="Run all pipeline steps")
group.add_argument("--narration", action="store_true", help="Run only Narration Generation")
group.add_argument("--qa", action="store_true", help="Run QA Generation (CBF → CSRD → CSPU)")
group.add_argument("--mcq", action="store_true", help="Run MCQ Generation (CBF → CSRD → CSPU)")
args = parser.parse_args()

# Load main pipeline configuration
try:
    with open("../../configs/video_qa_pipeline.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except Exception as e:
    logging.error(f"Failed to load video_qa_pipeline.yaml: {e}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("🚀 Starting Video QA Processing Pipeline...")

def run_script(script_name, config_path):
    """Run a python script with given config file"""
    cmd = ["python", script_name, "--config", config_path]
    logging.info(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logging.info(f"✅ {script_name} completed successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ {script_name} failed with error")

def main():
    config_dir = "../../configs"

    # Step 1: Narration
    if args.all or args.narration:
        logging.info("=== Step 1: Narration Generation ===")
        run_script("video_narration_generator.py", 
                   os.path.join(config_dir, "narration_config.yaml"))

    # Step 2: QA Generation (CBF → CSRD → CSPU)
    if args.all or args.qa:
        logging.info("=== Step 2: QA Generation (CBF → CSRD → CSPU) ===")
        for variant in config['qa']['variants']:
            logging.info(f"--- Running QA {variant.upper()} ---")
            config_path = os.path.join(config_dir, f"qa_config_{variant}.yaml")
            run_script("qa_generator.py", config_path)

    # Step 3: MCQ Generation (CBF → CSRD → CSPU)
    if args.all or args.mcq:
        logging.info("=== Step 3: MCQ Generation (CBF → CSRD → CSPU) ===")
        for variant in config['mcq']['variants']:
            logging.info(f"--- Running MCQ {variant.upper()} ---")
            config_path = os.path.join(config_dir, f"mcq_config_{variant}.yaml")
            run_script("mcq_generator.py", config_path)

    logging.info("🎉 Video QA Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
