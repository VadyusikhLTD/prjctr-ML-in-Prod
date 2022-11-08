from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    conv1channels_num: int = field(
        default=20,
        # metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    conv2channels_num: int = field(
        default=20,
        # metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    final_activation: Optional[str] = field(
        default=None,
        # metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    save_model: Optional[bool] = field(
        default=True,
        # metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
