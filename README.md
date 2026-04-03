# CLPsych 2026 Shared Task - Task 1

This directory contains the submission code and models for **Task 1** of the CLPsych 2026 Shared Task. 
Task 1 focuses on predicting the ABCD taxonomy of self-state labels (Adaptive and Maladaptive states) and tracking regression/progression across patient timelines based on post texts.

## Directory Structure & Approaches

The workspace is divided into several subdirectories representing our different modeling approaches and ablations:

* **`model1/`** & **`model2/`**: 
  Post-level taxonomy classification using a fine-tuned **Qwen2** backbone. These models take an individual post's text and directly predict both the taxonomy sub-elements (via linear classification heads mapping to 32 dimensions) and the presence score (regression head).
  
* **`model3/`**: 
  An Instruction-Tuning / Prompt-based approach (`Predict_Instruct.py`) leveraging Qwen2 to extract taxonomy self-states directly in a generative or zero/few-shot formatted structure.

* **`model4/`**: 
  A Timeline-level sequence modeling approach. It processes a user's entire timeline chronologically using a **BiLSTM Context Encoder**. This model is trained in two stages:
  1. **Stage 1**: Train the encoder along with a `TaxonomyClassifier` to predict taxonomy subelements (switches, escalation, elements).
  2. **Stage 2**: Freezes the sequence encoder and trains a `PresenceRegressor` to predict the numeric presence score (1-5) for each state.

* **`Contrastive_Learning/`**: 
  Trains a lightweight **DistilBERT** model as a **Contrastive Self-State Encoder**. It uses a multi-label supervised contrastive loss combined with BCE to map post text to embeddings that closely reflect the ABCD taxonomy self-state labels. These embeddings can be concatenated to generic post embeddings to enrich downstream timeline models.

## Generic Usage

Each model subdirectory essentially follows a similar execution interface consisting of standard scripts such as:

* `train.py`: Main training loops.
* `test.py` / `predict.py`: Run inference on a test/validation dataset.
* `evaluate_task1.py`: Computes the official CLPsych 2026 metrics for Task 1 formatting.
* `validate_submission.py`: Schema validation script.
* `code.ipynb`: Jupyter Notebooks providing end-to-end interactive code walkthroughs and experiments.

### Requirements & Setup
Code relies on standard transformers, PyTorch, and bitsandbytes for quantized Qwen inference.

_Note: The `dataset_cache` and static word embedding files (`wiki-news-300d-1M.vec`) are stored globally inside the project directories._
