# MDCure: A Scalable Pipeline for Multi-Document Instruction-Following

**MDCure** is an effective and scalable approach for synthetic data generation and curation tailored for multi-document (MD) instruction-tuning. This repository provides datasets, models, and source code to reproduce MDCure.


<p align="center">
  <a href="https://huggingface.co/collections/yale-nlp/mdcure-6724914875e87f41e5445395" style="display:inline-block;background-color:#4CAF50;color:white;padding:10px 20px;text-align:center;text-decoration:none;font-size:16px;border-radius:5px;margin-right:10px;">🤗 HF Repo</a> • <a href="https://aclanthology.org/2025.acl-long.1418/" style="display:inline-block;background-color:#2196F3;color:white;padding:10px 20px;text-align:center;text-decoration:none;font-size:16px;border-radius:5px;">📄 Paper</a>
</p>


## Table of Contents
- [🗂  Datasets](#datasets) & [ ⚙️  Models](#models)
- [🛠 MDCure Dataset Construction](#construction)
  <!-- - [📑 Source Data Preparation](#dataprep)
  - [✏️ Generation Phase](#generation)
  - [🔍 Filtering Phase](#filtering) -->
- [🖥️ MDCure Instruction Tuning](#it)
- [📊 Evaluation](#evaluation)
- [📝 Citation](#citation)

<a name="datasets"></a> 

## 🗂 Datasets
All size variations of our full MDCure datasets are available on HuggingFace. You can download and use the data via HuggingFace datasets:

| Dataset                   | HuggingFace Repo    | Description                  |
|---------------------------|---------------------|------------------------------|
| MDCure-12k     | [🤗 HF Repo](https://huggingface.co/datasets/yale-nlp/MDCure-12k) | Multi-document instruction dataset of size 12K, filtered using MDCureRM |
| MDCure-36k     | [🤗 HF Repo](https://huggingface.co/datasets/yale-nlp/MDCure-36k) | Multi-document instruction dataset of size 36K, filtered using MDCureRM |
| MDCure-72k     | [🤗 HF Repo](https://huggingface.co/datasets/yale-nlp/MDCure-72k) | Multi-document instruction dataset of size 72K, filtered using MDCureRM |


### Sample Dataset Entries

The table below contains example instances of the final MDCure instruction samples. Each dataset consists of multi-document instruction-answer pairs designed to simulate complex real-world information synthesis tasks. Each entry contains a comprehensive instruction, a context composed of multiple documents, and a well-formed answer.

| Instruction | Context | Answer |
|-------------|---------|--------|
| How did the combination of weather conditions, game dynamics, and viewership trends contribute to the overall perception of the Falcons-Eagles season opener, and what implications might this have for the NFL's ratings going forward? Respond with 3-4 sentences. | In what had to be a disappointing night for both NBC and the NFL, the Falcons and Eagles played a fairly ugly game after a lengthy weather delay. That's not how the league wanted to kick off the regular season, and while it got sort of exciting at the end, the overall recipe was a bad one for ratings. That was confirmed today, as the overnights came in... | The season opener between the Atlanta Falcons and Philadelphia Eagles faced a significant weather delay that postponed the game's start until 9:05 p.m. ET, leading to a lackluster atmosphere and frustration among fans eager for the NFL to return. The match itself was characterized by a dismal performance, marked by a total of 26 penalties compared to only 30 points scored, which caused offensive excitement to dwindle, especially... |
| What happened in CHAMPAIGN regarding Lovie Smith and the 2019 defense improvements? Respond with 1-2 sentences. | CHAMPAIGN - Lovie Smith knows his defense has to take a significant step forward in 2019. After defensive coordinator Hardy Nickerson left during the fall due to health concerns, Smith took over as the primary play-caller. This offseason, he didn't hire a defensive coordinator... | Lovie Smith took over as the defensive play-caller for 2019 after the previous coordinator, Hardy Nickerson, stepped down due to health concerns. Smith decided not to hire a replacement and instead took on the responsibilities himself. The defense has shown significant improvement in the spring practices. |

<a name="models"></a> 

## ⚙️ Models
<!-- We will soon release our **MDCure'd models**, instruction-tuned from various base models to exhibit superior performance on multi-document tasks, as well as **MDCureRM**, our custom reward model designed specifically for the MD setting to filter and select high-quality MD instruction data in a cost-effective way. -->
We release our **MDCure'd models**, instruction-tuned from various base models to exhibit superior performnace on multi-document tasks, as well as **MDCureRM**, our custom reward model designed specifically for the MD setting to filter and select high-quality MD instruction data in a cost-effective way.

| Model                     | Huggingface Repo    | Description                  |
|---------------------------|---------------------|------------------------------|
| MDCureRM  | [🤗 HF Repo](https://huggingface.co/yale-nlp/MDCureRM) | Multi-objective reward model to filter MD instruction data more cheaply and effectively than GPT-3.5-Turbo. |
| MDCure-FlanT5-Base       | [🤗 HF Repo](https://huggingface.co/yale-nlp/MDCure-FlanT5-Base) | **FlanT5-Base** fine-tuned with MDCure-72k, optimized for MD tasks.           |
| MDCure-FlanT5-Large      | [🤗 HF Repo](https://huggingface.co/yale-nlp/MDCure-FlanT5-Large) | **FlanT5-Large** fine-tuned with MDCure-72k, optimized for MD tasks.          |
| MDCure-Qwen2-1.5B-Instruct    | [🤗 HF Repo](https://huggingface.co/yale-nlp/MDCure-Qwen2-1.5B-Instruct) | **Qwen2-1.5B-Instruct** fine-tuned with MDCure-72k, optimized for MD tasks.   |
| MDCure-Qwen2-7B-Instruct      | [🤗 HF Repo](https://huggingface.co/yale-nlp/MDCure-Qwen2-7B-Instruct) | **Qwen2-7B-Instruct** fine-tuned with MDCure-72k, optimized for MD tasks.     |
| MDCure-LLAMA3.1-8B-Instruct   | [🤗 HF Repo](https://huggingface.co/yale-nlp/MDCure-LLAMA3.1-8B-Instruct) | **LLAMA3.1-8B-Instruct** fine-tuned with MDCure-72k, optimized for MD tasks.  |
| MDCure-LLAMA3.1-70B-Instruct  | [🤗 HF Repo](https://huggingface.co/yale-nlp/MDCure-LLAMA3.1-70B-Instruct) | **LLAMA3.1-70B-Instruct** fine-tuned with MDCure-72k, optimized for MD tasks. |


<a name="construction"></a> 

## 🛠 MDCure Dataset Construction
The MDCure dataset is constructed through a two-phase process:

1. **Generation Phase**: Zero-shot prompt templates are used to generate complex, cross-text instructions from related documents.
2. **Filtering Phase**: The generated instructions are filtered by MDCureRM, a multi-objective reward model, to ensure quality and diversity.

Below is a visual representation of the MDCure pipeline:

![MDCure Dataset Construction Diagram](./paper_figures/mdcure_dataset_construction.png)

<a name="dataprep"></a> 

### 📑 0. Source Data Preparation
- We use the NewSHead dataset as the source for sets of (related) context documents. The pre-processed and cleaned NewSHead dataset can be downloaded from [here](https://storage.googleapis.com/primer_summ/newshead_data.tar.gz). After downloading the data, unzip the file to a new folder named `./0_source_data_preparation/newshead`.
- To extract the clusters of articles from the NewSHead dataset, create the `./0_source_data_preparation/base_articles` directory and then run the following command:
```bash
python ./0_source_data_preparation/extract_source_articles.py
```
- In addition to the article clusters, we utilize pairs of snippets ranging from 1-3 sentences in length selected from different documents within each cluster to perform instruction data generation with a portion of the MDCure generation prompt templates. To generate relevant pairs of snippets for each cluster in NewSHead and save to the `./0_source_data_preparation/base_snippet_pairs` directory, run:
```bash
python ./0_source_data_preparation/save_snippet_pairs.py
```

<a name="generation"></a> 

### ✏️ 1. Generation Phase 
- During the Generation phase of MDCure, we employ two types of prompt templates (General & Style-Specific) to encourage a generator model to produce a diverse set of instruction data. The specific prompt templates we utilize to elicit relevant multi-document instruction-answer pairs based on the provided context documents and/or text snippets can be found in `./mdcure_prompt_templates`. These are described further below.
- We use GPT-3.5-Turbo as our generator model, but other powerful instruction-following LLMs may be substituted in its place. Be sure to set your OpenAI API key for use with the code.
- Please note that depending on the number of instructions being generated, this process can be slow as it relies on numerous calls to the OpenAI API. To avoid issues due to interruptions to the program execution, we define a save interval of every 10 instructions, such that after producing 10 samples the results will be saved to a distinct `.pt` file. The save interval can be adjusted as desired in `./1_mdcure_generation/generate_general.py` and `./1_mdcure_generation/generate_style_specific.py`.

#### Environment Setup
Install the required packages:

```bash
pip install -r ./1_mdcure_generation/gen_requirements.txt
```

#### Generation with Prompt Templates A-F (General Templates)
General templates broadly ask the generator model to produce an instruction-answer pair adhering to a certain answer length, with emphasis placed on creation of instructions that cannot be answered if any one context document is removed; we employ various phrasings of this specification to obtain prompts A-F. Further variants are captured by prompts G-N, described below.

To generate instruction-answer pairs using general prompts A-F, use `./1_mdcure_generation/generate_general.py`. This file can be run with several combinations of arguments — for basic usage, you can run:
```bash
# Create generation prompts for training set
python ./1_mdcure_generation/generate_general.py --only_generate_prompts --num_prompts_to_generate=<desired_num_train_generations> --use_train --prompt_num=<prompt_num>

# Create generation prompts for validation set
python ./1_mdcure_generation/generate_general.py --only_generate_prompts --num_prompts_to_generate=<desired_num_valid_generations> --use_valid --prompt_num=<prompt_num>

# Use prompts to elicit candidate instructions
python ./1_mdcure_generation/generate_general.py --use_existing_prompts --given_prompt_dir=./1_mdcure_generation/general_instructions/prompt_<prompt_id>/prompts
```
The resulting instructions will be saved to a folder path of the form:
```bash
./1_mdcure_generation/generations_general/prompt_<prompt_id>/instructions
```
And the prompts used to obtain these instructions will be saved to:
```bash
./1_mdcure_generation/generations_general/prompt_<prompt_id>/prompts
```

#### Generation with General Prompt Templates G-N & Style-Specific Templates

Style-Specific templates employ a single phrasing while incorporating varying combinations of constraints regarding instruction complexity (e.g., analysis, recall), type (e.g., paraphrasing, inference), style (e.g., imperative, interrogative), and output length (e.g., 3-4 words, 2 sentences), with pre-defined options for each specification category. This helps enforce constraints on diversity of the generated candidate data. We consider 4 characteristics, 4 types, and 3 styles of instruction, in combination with 8 possible answer lengths. Any number of context documents can be considered for generation in these prompt templates. Further details can be seen in `./mdcure_prompt_templates`.

To generate instruction-answer pairs using general prompts G-N or style-specific prompts, use `./1_mdcure_generation/generate_style_specific.py`. This file can be run with several combinations of arguments — for basic usage, you can run:
```bash
# For general templates G-N
python ./1_mdcure_generation/generate_style_specific.py  --use_train  --model=gpt-4o-mini  --num_prompts_to_generate=<desired_number_of_generations> --num_docs_to_prompt_with=<desired_number_of_docs_in_context>  --straightforward_prompt  --prompt_num=<prompt_num>

# For style-specific templates
python ./1_mdcure_generation/generate_style_specific.py  --use_train  --model=gpt-4o-mini  --num_prompts_to_generate=<desired_number_of_generations>  --template_prompt  --num_docs_to_prompt_with=<desired_number_of_docs_in_context>  --char_idx=<complexity_specification_index>  --type_idx=<type_specification_index>  --style_idx=<style_specification_index>  --length_idx=<output_length_specification_index>  --cluster_start_multiplier=<idx>
```
The resulting instructions will be saved to a folder path of the form:
```bash
./1_mdcure_generation/generations_style_specific/<template_description>/instructions
```
And the prompts used to obtain these instructions will be saved to:
```bash
./1_mdcure_generation/generations_style_specific/<template_description>/prompts
```

#### Extraction of Generated Instruction-Answer Pair Candidates
To extract the generated instruction-answer pairs from the raw outputs by the generator model, we use `./1_mdcure_generation/extract_general.py` and `./1_mdcure_generation/extract_style_specific.py`. Sample usages are as follows:
```bash
# For general templates A-F
python ./1_mdcure_generation/extract_general.py --input_dir=./1_mdcure_generation/generations_general/prompt_<prompt_id> --prompt_num=<prompt_num>

# For general templates G-N & style-specific templates
python ./1_mdcure_generation/extract_style_specific.py --input_dir=./1_mdcure_generation/generations_style_specific/<template_specification>
```

#### Addition of Length Direction
For all samples we append an additional sentence or phrase indicating the length of the expected answer output, in terms of the number of words or sentences to produce. For data generated using prompts G-N or style-specific prompts, the length direction is automatically added during the extraction step above. For data generated using prompts A-F, this is done with the `./1_mdcure_generation/add_length_direction.py` file, which can be used as such:
```bash
python ./1_mdcure_generation/add_length_direction.py --data_json_dir=<path_to_directory_with_generated_instruction_jsons>
```

<a name="filtering"></a> 

### 🔍 2. Filtering Phase
During the Filtering phase of MDCure, we train and employ a fine-grained, _MD-specific_ reward model, **MDCureRM**, to evaluate each instruction-answer pair based on six different criteria. These criteria capture both the overall quality of the instruction-response pairs and their effectiveness in handling multi-document content and are shown below:

#### MDCureRM Scoring Criteria

| Criterion | Description |
| -------- | ----------- |
| Context Integration                | How well does the instruction synthesize information from multiple documents? |
| Inter-Document Relationships       | Does the instruction help understand relationships between documents? |
| Complexity                         | Does the instruction challenge critical thinking and analysis skills over multiple sources? |
| Relevance                          | Is the instruction aligned with the document content? |
| Coherence & Factuality             | Is the response logical, factual, and well-supported by the documents? |
| Creativity                         | How diverse or creative is the instruction in question type and format? |

MDCureRM is used in a RM-as-a-Judge fashion to filter the generated instruction set and yield high-quality training data more cheaply and effectively than GPT-3.5-Turbo. We walk through the MDCureRM training and inference process below.

### Building MDCureRM

#### Environment Setup
Install the requirements with pip: 
```bash
pip install -r ./2_mdcure_filtering/mdcurerm/rm_requirements.txt
```

#### Training Data Preparation

To create fine-grained reward training data for the MDCureRM, follow these steps:

1. **Generate Candidate Instructions**:
   - Use the `get_rm_candidate_instructions.py` script to generate instruction samples from models such as Mistral-7B Instruct or GPT-4o-mini. Alternatively, you can build your own generation pipeline using the instruction generation prompts. These generated instructions will serve as input for the reward model training pipeline.
   - ```bash
     python ./2_mdcure_filtering/mdcurerm/get_rm_candidate_instructions.py --rubric <rubric> --num_sample 2000 --file_path <path_to_context_docs> --openai_key <openai_api_key> --mistral_key <mistral_api_key>
     ```

2. **Prepare and Finalize Training Data**:
   - Once the candidate instructions are generated, use GPT-4 or a similar model to assign ratings based on criteria such as relevance, coherence, and quality. These ratings will be used to train the fine-grained reward model.
   - ```bash
     python ./2_mdcure_filtering/mdcurerm/get_rm_training_data.py
     ```

#### Training MDCureRM
To train the MDCure Reward Model (RM) based on 1-layer regression, follow these steps:

1. **Input Data:**
   - The expected input data should be in JSONL format and located as `./data/parsed_ratings.jsonl`. This file contains the parsed training data required for the reward model.

2. **Running the Training:**
   - To initiate the training process, execute the `train_rm.py` script. The script is designed to use the `parsed_ratings.jsonl` data and will set up a 1-layer regression model appended to the end of the base LLM, which is initialized with Bradley-Terry weights, based on the input.

3. **Example Command:**
   ```bash
   torchrun --nproc_per_node=8 ./2_mdcure_filtering/mdcurerm/train_rm.py \ 
      --output_dir=<model_save_path> \
      --per_device_train_batch_size=1 \
      --per_device_eval_batch_size=1 \
      --num_train_epochs=4 \
      --gradient_accumulation_steps=8 \
      --learning_rate=1e-4 \
      --report_to="wandb" \
      --optim="adamw_8bit" \
      --logging_steps=5 \
      --eval_strategy="steps" \
      --evaluation_strategy="steps" \
      --save_strategy="steps" \
      --eval_steps=125 \
      --save_steps=125 \
      --fp16=True \
      --lr_scheduler_type="linear" \
      --warmup_ratio=0.1 \
      --remove_unused_columns=False \
      --save_total_limit=10 \
      --push_to_hub=False \
      --label_names="rewards" \
      --wandb_run_name="real_1e-4" \
      --preference_data_path="./data/parsed_ratings.jsonl" \
      --base_model_name="sfairXC/FsfairX-LLaMA3-RM-v0.1"
   ```
#### Using MDCureRM
The `score_and_filter.py` script is designed to score instructions using a pretrained reward model. It evaluates instruction data against multiple scoring criteria and filters out low-quality instructions based on the threshold set by the user. The model applies a weighted average of six scoring criteria to predict instruction quality.

This script can be used to score instruction samples with the following command:

```bash
python ./2_mdcure_filtering/score_and_filter.py \
    --input_dir=<path_to_instruction_jsons> \
    --scoring_model_path=./reward_modeling/models/reward_model \
    --prompt_num=0 \
    --score_general \
    --score_style_specific
```

#### Script Arguments
  - `--input_dir`: Directory containing the input JSON files for training and validation, created during the Generation phase.
  - `--scoring_model_path`: Path to MDCureRM.
  - `--prompt_num`: Prompt ID for general template-based samples.
  - `--score_general`: Flag to indicate scoring of general template-based samples.
  - `--score_style_specific`: Flag to indicate scoring of style-specific template-based samples.

#### Important Notes
- **HF API Key**: Ensure that your Hugging Face (HF) API key is set at the top of the script to access the LLAMA model. This is required for model loading.
  
- **Scoring Mechanism**: MDCureRM generates a vector of six floating-point values (one per criterion), which are scaled to the 1-5 range. These six scores are combined into an overall score using a weighted sum, with general quality criteria (Relevance, Coherence & Factuality, Creativity) weighted at $\frac{1}{9}$, and multi-document-specific criteria (Context Integration, Inter-Document Relationships, Complexity) weighted at $\frac{2}{9}$.

<a name="it"></a> 

## 🖥️ Instruction-Tuning

### Environment Setup
To set up the environment, install the necessary dependencies using the following command:

```bash
pip install -r ./instruction_tuning/it_requirements.txt
```

### Training

We utilize [Unsloth](https://github.com/unslothai/unsloth) to train models at a large scale (billion-parameter models). For detailed guidance on Unsloth, please refer to their official GitHub repository.

We use the `./instruction_tuning/train.py` to fine-tune various models, such as Flan-T5 and LLAMA models via instruction-tuning. The script supports multiple options for model architectures, such as Seq2Seq models (`T5`) or autoregressive models (`LLAMA`, `Qwen`), along with PEFT (Parameter Efficient Fine-Tuning) techniques like LoRA.

To train a model using Unsloth, for example, the LLAMA3.1-8B-Instruct model, you can use the following command:

```bash
torchrun --nproc_per_node=1 --rdzv_endpoint=127.0.0.1:46967 ./instruction_tuning/train.py \
    --model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" \
    --wandb_project_name=<proj_name> \
    --output_dir=./instruction_tuning/models \
    --lr=5e-5 \
    --lr_scheduler_type="linear" \
    --optimizer="adamW" \
    --warmup_ratio=0.15 \
    --weight_decay=0.01 \
    --save_total_limit=<save_limit> \
    --per_train_batch_size=4 \
    --per_eval_batch_size=4 \
    --grad_acc_steps=16 \
    --wandb_group=<group_name> \
    --num_train_steps=<train_steps> \
    --save_steps=<save_steps> \
    --eval_steps=<eval_steps> \
    --train_num_samples=<train_num_samps> \
    --valid_num_samples=<valid_num_samps> \
    --train_json_path=<train_json_path> \
    --valid_json_path=<valid_json_path> \
    --train_num_samples2=<train_num_samps2> \
    --valid_num_samples2=<valid_num_samps2> \
    --train_json_path2=<train_json_path2> \
    --valid_json_path2=<valid_json_path2> \
    --tokenization_batch_size=1 \
    --use_truncation \
    --use_unsloth \
    --use_chat_template \
    --choose_by_rank \
    --choose_by_rank2
```

#### Training Script Arguments

- **Model-specific Arguments**:
  - `--model_name`: Specify the pretrained model to fine-tune (e.g., `google/long-t5-tglobal-base`, `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`).
  - `--use_unsloth`: Activate this flag when training with Unsloth models.
  - `--use_peft`: Enable PEFT (e.g., LoRA) for large models with memory-efficient training.

- **Wandb Logging**:
  - `--wandb_project_name`: Define the Weights and Biases (Wandb) project to log results.
  - `--wandb_group`: Specify a group for the runs, helpful for tracking related experiments.

- **Training Configuration**:
  - `--lr`: Learning rate for training.
  - `--lr_scheduler_type`: Learning rate schedule (e.g., constant, linear).
  - `--optimizer`: Optimizer to use (e.g., `adamW` or `adafactor`).
  - `--grad_acc_steps`: Gradient accumulation steps.
  - `--save_steps` and `--eval_steps`: Number of steps between saving checkpoints and running validation scores.
  - `--num_train_steps`: Total number of training steps.
  - `--save_total_limit`: Max number of checkpoints to save.
  - `--per_train_batch_size` and `--per_eval_batch_size`: Per-device training and evaluation batch sizes.

- **Dataset-related Arguments**:
  - `--train_num_samples` and `--valid_num_samples`: Number of training and validation samples.
  - `--train_json_path` and `--valid_json_path`: Paths to the JSON files for training and validation datasets.
  - `--choose_by_rank`: Enable sampling based on scores (e.g., for MDCure datasets).
  
- **Tokenization**:
  - `--tokenization_batch_size`: Set batch size for tokenization. Must be 1 for MDCure datasets.
  - `--truncate_max_length`: Maximum truncation length when tokenizing long sequences for training.

<a name="evaluation"></a> 

## 📊 Evaluation

We list below the evaluation benchmarks that we used in our paper. For Multi-XScience and QMDSCNN, we utilize GPT-3.5-Turbo via the OpenAI API and Gemini 1.5 Pro via the Gemini API to obtain scores of results, respectively; for all other benchmarks we utilize their built-in evaluation metrics. 

[HotpotQA](https://hotpotqa.github.io/)

[WikiHop](https://qangaroo.cs.ucl.ac.uk/)

[Multi-XScience](https://github.com/yaolu/Multi-XScience)

[QMDSCNN](https://github.com/ramakanth-pasunuru/QmdsCnnIr)

[SEAM](https://seam-benchmark.github.io/)

[ZeroScrolls](https://www.zero.scrolls-benchmark.com/)


<a name="citation"></a> 

## 📝 Citation

If you find the content of this project helpful, please cite our paper as follows:
```bibtex
@inproceedings{liu-etal-2025-mdcure,
    title = "{MDC}ure: A Scalable Pipeline for Multi-Document Instruction-Following",
    author = "Liu, Gabrielle Kaili-May  and
      Shi, Bowen  and
      Caciularu, Avi  and
      Szpektor, Idan  and
      Cohan, Arman",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1418/",
    doi = "10.18653/v1/2025.acl-long.1418",
    pages = "29258--29296",
    ISBN = "979-8-89176-251-0",
    abstract = "Multi-document (MD) processing is crucial for LLMs to handle real-world tasks such as summarization and question-answering across large sets of documents. While LLMs have improved at processing long inputs, MD contexts still present unique difficulties, including management of inter-document dependencies, redundancy, and incoherent structures. To address this challenge, we introduce MDCure, a scalable and effective instruction data generation framework to enhance the MD capabilities of LLMs without the computational cost of pre-training or reliance on human-annotated data. MDCure generates high-quality synthetic MD instruction data over sets of articles via targeted prompts. We also introduce MDCureRM, a cost-effective, MD-specific reward model to score and filter generated data based on their training utility for MD settings. MDCure is compatible with open- and closed-source models in addition to policy optimization methods such as PPO, enabling even small open- source models to surpass proprietary LLMs as strong generators of high-quality MD instruction data without further data filtering. With MDCure, we fine-tune a wide variety of LLMs up to 70B parameters in size from the FlanT5, Qwen2, and LLAMA3.1 model families. Extensive evaluations on a wide range of MD and long-context benchmarks spanning various tasks and domains show MDCure consistently improves performance over pre-trained baselines and base models by up to 75.1{\%}."
}
```
