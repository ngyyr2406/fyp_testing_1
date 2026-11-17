"""
Model loading utilities for the FYP resume-job matching application.

Expected checkpoint layout:
model_outputs/
└── checkpoints/
    ├── tfidf_vectorizer.pkl
    ├── bert_base_uncased_checkpoint/
    ├── e5_finetuned_checkpoints/
    │   ├── e5_fold_1_checkpoint/
    │   ├── ...
    │   └── e5_fold_5_checkpoint/
    └── bge_finetuned_checkpoints/
        ├── bge_fold_1_checkpoint/
        ├── ...
        └── bge_fold_5_checkpoint/

The E5 and BGE fold checkpoints in this project are from selected Experiment 2.
"""

from pathlib import Path
import json
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


DEFAULT_BASE_DIR = Path("fyp code")


def resolve_project_paths(base_dir=None):
    """Resolve project, config and checkpoint paths."""
    if base_dir is None:
        # Works when running locally from the project root, or in Colab after changing BASE_DIR to the Google Drive project path.
        base_dir = Path("/content/drive/MyDrive/fyp code")
    else:
        base_dir = Path(base_dir)

    output_dir = base_dir / "model_outputs"
    checkpoint_dir = output_dir / "checkpoints"
    config_path = output_dir / "model_config.json"

    return {
        "base_dir": base_dir,
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
        "config_path": config_path,
    }


def _require_path(path, description):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n{path}\n"
            "Check your Google Drive mount and checkpoint directory."
        )
    return path


def load_model_config(paths):
    config_path = _require_path(paths["config_path"], "model_config.json")
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_tfidf(checkpoint_dir):
    """Load the fitted TF-IDF vectorizer saved in notebook Cell 22."""
    path = _require_path(
        checkpoint_dir / "tfidf_vectorizer.pkl",
        "TF-IDF vectorizer checkpoint",
    )
    return joblib.load(path)


def load_bert(checkpoint_dir, device):
    """Load the saved pretrained bert-base-uncased model and tokenizer."""
    path = _require_path(
        checkpoint_dir / "bert_base_uncased_checkpoint",
        "BERT checkpoint directory",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModel.from_pretrained(str(path), local_files_only=True)
    model.to(device)
    model.eval()
    return {"tokenizer": tokenizer, "model": model}


def load_sentence_transformer_folds(
    checkpoint_dir,
    model_folder_name,
    model_prefix,
    device,
    expected_folds=5,
):
    """
    Load all five saved fold checkpoints for the selected Experiment 2.
    All fold models are used as an embedding ensemble at inference time.
    """
    parent = _require_path(
        checkpoint_dir / model_folder_name,
        f"{model_prefix.upper()} checkpoint folder",
    )

    models = []
    for fold in range(1, expected_folds + 1):
        fold_path = _require_path(
            parent / f"{model_prefix}_fold_{fold}_checkpoint",
            f"{model_prefix.upper()} fold {fold} checkpoint",
        )
        model = SentenceTransformer(str(fold_path), device=str(device))
        model.eval()
        models.append(model)

    return models


def load_all_models(base_dir=None, device=None):
    """
    Load TF-IDF, BERT, E5 and BGE.

    E5/BGE each load their five saved Experiment 2 fold checkpoints.
    No model training is performed here.
    """
    paths = resolve_project_paths(base_dir)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    config = load_model_config(paths)
    checkpoint_dir = paths["checkpoint_dir"]

    models = {
        "device": device,
        "config": config,
        "tfidf": load_tfidf(checkpoint_dir),
        "bert": load_bert(checkpoint_dir, device),
        "e5": load_sentence_transformer_folds(
            checkpoint_dir,
            "e5_finetuned_checkpoints",
            "e5",
            device,
        ),
        "bge": load_sentence_transformer_folds(
            checkpoint_dir,
            "bge_finetuned_checkpoints",
            "bge",
            device,
        ),
    }

    return models
