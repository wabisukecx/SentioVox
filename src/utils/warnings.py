import warnings
from contextlib import contextmanager


@contextmanager
def suppress_warnings():
    """システム全体の警告を抑制するコンテキストマネージャー"""
    with warnings.catch_warnings():
        # Whisper関連の警告を抑制
        warnings.filterwarnings(
            "ignore",
            message="Failed to launch Triton kernels*",
            category=UserWarning
        )
        # torch.load関連の警告を抑制
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="You are using `torch.load`*"
        )
        # トークン化関連の警告を抑制
        warnings.filterwarnings(
            "ignore",
            message="Asking to truncate to max_length*"
        )
        yield