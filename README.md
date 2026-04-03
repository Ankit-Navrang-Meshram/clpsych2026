[cite_start]Here is the updated README file, now including the official challenge description and framework overview based on the CLPsych 2026 guidelines[cite: 3].

***

# CLPsych 2026 Shared Task - Task 1

This repository contains the submission code and models for **Task 1** of the CLPsych 2026 Shared Task. 

## Challenge Description & Overview

[cite_start]The CLPsych 2026 Shared Task builds upon the longitudinal modeling paradigms introduced in previous years (2022, 2025), which focused on detecting changes in individuals' mental states and well-being as reflected in social media timelines[cite: 4]. [cite_start]The 2026 task further advances this work by emphasizing the identification of key self-state elements that lead up to dynamic mental state changes over time[cite: 5]. 

[cite_start]The task is grounded in the **MIND framework**, which conceptualizes self-states as structured combinations of four components (ABCD)[cite: 7]:
* [cite_start]**Affect (A):** Emotional tone or mood[cite: 11].
* [cite_start]**Behavior (B-S, B-O):** Actions or tendencies directed inward (toward self) or outward (toward others)[cite: 12].
* [cite_start]**Cognition (C-S, C-O):** Beliefs, interpretations, and appraisals toward the self and others[cite: 13].
* [cite_start]**Desire (D):** Motivations, needs, wishes, and expectations[cite: 14].

[cite_start]The core objective of the shared task is to develop computational models capable of identifying these adaptive and maladaptive self-state dimensions, characterizing their presence, and summarizing self-state sequences leading to critical mental health changes across social media timelines[cite: 9].

## Task Breakdown

Our models are designed to tackle the two primary subtasks of Task 1:

### Task 1.1: Dominant ABCD Subelement Identification
[cite_start]Given a post, the objective is to identify which predefined ABCD subelements are meaningfully expressed[cite: 37]. [cite_start]A self-state is either adaptive or maladaptive[cite: 29]. [cite_start]If multiple subelements within the same element and valence are expressed, the models must predict only the single most dominant, central, and emphasized subelement[cite: 41].

### Task 1.2: Self-State Presence Rating
[cite_start]For each identified adaptive and maladaptive self-state, the models estimate the psychological centrality and experiential influence of that state on a scale of 1 to 5[cite: 55, 65]. [cite_start]A score of 1 indicates the state is not present, while a 5 indicates it highly shapes and defines the overall experience[cite: 61]. [cite_start]This is treated as a regression task evaluated using the RMSE metric[cite: 70].

---

## Directory Structure & Modeling Approaches

Our workspace is structured into several subdirectories representing different modeling architectures and ablations:

* **`model1/` & `model2/`:** Post-level taxonomy classification utilizing a fine-tuned **Qwen2** backbone. These models take an individual post's text as input and directly predict the taxonomy sub-elements (via linear classification heads) and the presence score (via a dedicated regression head).
* **`model3/`:** An Instruction-Tuning / Prompt-based approach (`Predict_Instruct.py`). This method leverages the generative capabilities of Qwen2 to extract taxonomy self-states directly in a structured format.
* **`model4/`:** A Timeline-level sequence modeling approach designed to capture temporal context across multiple posts. It processes a user's entire timeline chronologically using a **BiLSTM Context Encoder**. Training is conducted in two distinct stages:
    1.  **Stage 1:** Jointly trains the BiLSTM encoder and a `TaxonomyClassifier` to predict taxonomy subelements (Task 1.1) and timeline events (Switch & Escalation).
    2.  **Stage 2:** Freezes the BiLSTM sequence encoder and trains a separate `PresenceRegressor` to predict the numeric presence score (1-5) for each state (Task 1.2).
* **`Contrastive_Learning/`:** Trains a lightweight **DistilBERT** model as a dedicated **Contrastive Self-State Encoder**. It maps post text to dense embeddings that mathematically reflect the structural similarity of their ABCD taxonomy self-state labels.

---

## Generic Usage

Each model subdirectory adheres to a standardized execution interface containing the following core scripts:

* **`train.py`**: Executes the main training loops.
* **`test.py` / `predict.py`**: Runs inference on the test datasets to generate the required `task1_pred.json` output format.
* [cite_start]**`evaluate_task1.py`**: Computes the official CLPsych 2026 metrics for Task 1. This includes calculating Precision, Recall, and Macro/Micro F1 for element presence and subelement classification, as well as MAE, RMSE, Quadratic Weighted Kappa (QWK), and Spearman correlation for the presence ratings[cite: 83, 84, 95].
* [cite_start]**`validate_submission.py`**: Validates the output JSON schema against the official Shared Task requirements, ensuring no post text is accidentally included in the final submission[cite: 110, 111].
* **`code.ipynb`**: Jupyter Notebooks providing interactive, end-to-end code walkthroughs.

## Requirements & Setup

The codebase relies on modern deep learning libraries, specifically tailored for efficient LLM inference:

* `torch` (PyTorch)
* `transformers` (Hugging Face)
* `peft` (Parameter-Efficient Fine-Tuning)
* `bitsandbytes` (For 4-bit/8-bit quantized Qwen inference)
* `sentence-transformers`

**Note on Static Assets:** The `dataset_cache/` and static word embedding files (such as `wiki-news-300d-1M.vec`) are stored globally outside the individual model subdirectories and must be present in the expected data path before running the embedder scripts.

## Submission Formatting Note


[cite_start]As per the task guidelines, our inference scripts output a single `task1_pred.json` file containing a JSON array of per-post prediction objects[cite: 133]. [cite_start]To comply with privacy requirements, all original post text fields (e.g., "post", "text", "body") are stripped from the final JSON outputs prior to submission[cite: 131].