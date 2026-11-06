# K9Bench Dataset Generation Pipeline

---

## Repository Structure

```
K9Bench_Source_Code/
├── configs/                          # All YAML configuration files
│   ├── mcq_config_grp_1.yaml
│   ├── mcq_config_grp_2.yaml
│   ├── mcq_config_grp_3.yaml
│   ├── mcq_speaker_info_removal.yaml
│   ├── narration_config.yaml
│   ├── qa_config_grp_1.yaml
│   ├── qa_config_grp_2.yaml
│   ├── qa_config_grp_3.yaml
│   ├── speaker_information_retrieval.yaml
│   ├── video_collection_pipeline.yaml
│   └── video_qa_pipeline.yaml
├── data/
│   ├── meta_info/
│   │   ├── prompts/                  # Prompts for each pipeline stage
│   │   └── video_ids.txt
│   └── video_data/
│       ├── filtered_videos/
│       ├── temp/frames/
│       └── videos/
├── modules/
│   ├── bias_mitigation/
│   ├── qa_generation_pipeline/
│   └── video_collection_pipeline/
└── tools/
    ├── human_evals/
    └── human_review/
```

---



## Setup

Install the required dependencies:

```bash
 pip install -r requirements_pipeline.txt
```

---



## Pipeline Overview

```
1. Video Collection
2. Human Verification
3. QA/MCQ Generation
4. Bias Mitigaton
```

---



## Configuration Files

All pipeline stages are configured via YAML files in the `configs/` directory:

> **Note:** Before running the pipeline, review the configuration files and update the required parameters (e.g., download mode, search queries or initial video IDs, and Hugging Face dataset names where applicable).


| Config File                                                             | Stage                               |
| ----------------------------------------------------------------------- | ----------------------------------- |
| `video_collection_pipeline.yaml`                                        | Video Collection                    |
| `narration_config.yaml`                                                 | Narration Generation                |
| `qa_config_grp_1.yaml``qa_config_grp_2.yaml``qa_config_grp_3.yaml`    | QA Generation                       |
| `mcq_config_grp_1.yaml``mcq_config_grp_2.yaml``mcq_config_grp_3.yaml` | MCQ Generation                      |
| `speaker_information_retrieval.yaml`                                    | Bias Mitigation — Speaker Retrieval |
| `mcq_speaker_info_removal.yaml`                                         | Bias Mitigation — Speaker Removal   |

---



## Stage 1: Video Collection Pipeline 

> Paper : Section 3.1

Navigate to the pipeline directory and configure settings before running.

```bash
cd dataset_creation_pipeline/modules/video_collection_pipeline
```

Edit `configs/video_collection_pipeline.yaml` to set your parameters.

> **Note:** To use your own YouTube API keys and input source, update the following fields in the configuration file:
>
> - `key_file`: Path to your YouTube API keys file (e.g., `../../data/meta_info/youtube_api_keys.txt`).
> - `retry_delay` and `max_delay`: Configure API retry behaviour if needed.
> - `input.source`: Specify either a file containing video IDs (e.g., `../../data/meta_info/video_ids.txt`) or a search query (e.g., `"dog barking"`).
> - `input.query_max_results`: Set the maximum number of videos to retrieve when using a search query.

**Run the full pipeline:**

```bash
python video_collection_pipeline.py --all
```

**Or run individual steps:**

```bash
python video_collection_pipeline.py --collect_video   # Collect video metadata
python video_collection_pipeline.py --download_video  # Download videos
python video_collection_pipeline.py --video_filter    # Filter videos
```

> Output videos and metadata are saved to the `data/` folder.

---



## Stage 2: Human Verification

> Paper: Appendix B.2

After video collection, validate the collected videos using the human review tool.

```bash
cd dataset_creation_pipeline/tools/human_review/collected_video_validation
node server.js path_to_txt_file.txt
```

---



## Stage 3: QA / MCQ Generation Pipeline

> Paper: Section 3.3

Navigate to the QA generation pipeline directory:

```bash
cd dataset_creation_pipeline/modules/qa_generation_pipeline
```

**Run the full pipeline:**

```bash
python video_qa_pipeline.py --all
```

**Or run individual steps:**


| Flag          | Description                              |
| ------------- | ---------------------------------------- |
| `--narration` | Generate video narrations only           |
| `--qa`        | Run QA Generation (GRP 1→ GRP 1→ GRP 3)  |
| `--mcq`       | Run MCQ Generation (GRP 1→ GRP 1→ GRP 3) |


```bash
python video_qa_pipeline.py --narration
python video_qa_pipeline.py --qa
python video_qa_pipeline.py --mcq
```

> Group 1 (GRP 1) includes posture analysis and action sequence; Group 2 (GRP 2) includes context analysis and cause-effect analysis and Group 3 (GRP 3) includes interaction analyses. (Paper: Appendix D1)
>
> After MCQ generation, concatenate the output files with their category names before proceeding to Bias Mitigation.

---

## Stage 4: Bias Mitigation

> Paper: Section 3.4

This stage has three steps: running deaf-blind LLM evaluation, retrieving speaker information, and removing speaker info from MCQs.

### Step 4.1 — Deaf-Blind LLM Filtering

Answers every MCQ with three text-only LLMs that never see the video. A question
the majority of them get right is solvable from language alone, so it is dropped.

This is two steps: run each model, then take the vote. Set `dataset_name` in
`configs/deafblind_eval.yaml` to the QA pool *before* filtering — this stage is
what produces the filtered set.

```bash
cd dataset_creation_pipeline/modules/bias_mitigation

# 1. One run per model (GPU). Shard with --num_workers/--worker_id.
for M in deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
         Qwen/Qwen3-32B \
         mistralai/Mistral-Small-3.2-24B-Instruct-2506; do
    python deafblind_eval.py --config ../../configs/deafblind_eval.yaml --model_name "$M"
done

# 2. Majority vote -> IDs to remove
OUT=../../data/intermediary_files/bias_mitigation/deafblind/outputs
python deafblind_results.py \
    --model "$OUT/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/*.json" \
    --model "$OUT/Qwen/Qwen3-32B/*.json" \
    --model "$OUT/mistralai/Mistral-Small-3.2-24B-Instruct-2506/*.json" \
    --output example_ids_majority_voted.txt
```

`example_ids_majority_voted.txt` lists the question IDs to drop. `--min_votes`
defaults to 2 (majority of three).

> **Note:** Generation is sampled, not greedy (temperature 0.7, or 0.15 for
> Mistral, as in the reported runs), so rejection rates vary by a point or two
> between runs.



### Step 4.2 — Speaker Information Retrieval

Identifies speaker or identity cues present in the MCQs.

```bash
cd dataset_creation_pipeline/modules/bias_mitigation
python speaker_information_retrieval.py --config ../../configs/speaker_information_retrieval.yaml
```

> Note: Make sure to change the huggingface dataset name after deafblind filtering in config file `configs/mcq_speaker_info_removal`



### Step 4.3 — Speaker Information Removal

Removes identified speaker/identity information from MCQs to produce the final unbiased question set.

```bash
cd dataset_creation_pipeline/modules/bias_mitigation
python mcq_speaker_info_removal.py --config ../../configs/mcq_speaker_info_removal.yaml
```

> You now have a final, high-quality, bias-mitigated MCQ dataset ready for evaluation.

---

