# stdlib
from typing import Any, Callable, Optional
from pathlib import Path
# thirdpartylib
import joblib # pyright: ignore[reportMissingTypeStubs]
import torch
import torch.nn as nn
from keras import Model # pyright: ignore[reportMissingTypeStubs]
# projectlib
from pv_inverter_modeling.utils.typing import Address, Verbosity
from pv_inverter_modeling.utils.paths import validate_address
from pv_inverter_modeling.utils.logging import Logger

def save_model(
        model: Any,
        model_name: str,
        output_dir: Address,
        *,
        verbosity: Verbosity = 0,
        log_address: Address = Path.cwd(),
        write_log: bool = False,
    ) -> None:
    """
    Persist a trained model to disk with lightweight logging.

    This utility supports saving both PyTorch models and generic
    Python models. PyTorch `nn.Module` instances are saved via their
    `state_dict`, while all other model objects are serialized using
    `joblib`.

    Logging is optional and verbosity-controlled, making this function
    suitable for scripts, pipelines, and batch jobs without requiring
    a full logging framework.

    Parameters
    ----------
    model : Any
        Trained model object to persist. If the model is an instance of
        `torch.nn.Module`, its state dictionary is saved. Otherwise, the
        model is serialized using `joblib`.
    model_name : str
        Base name used to construct the output filename.
    output_dir : Address
        Directory in which the model file will be saved.
    verbosity : Verbosity, default 0
        Verbosity threshold for logging. Higher values emit more 
        messages.
    log_address : Address, default Path.cwd()
        Directory where the log file is written if logging to file is
        enabled.
    write_log : bool, default False
        If True, log messages are written to a log file. If False, log
        messages are printed to stdout.

    Notes
    -----
    - PyTorch models are saved as `<model_name>_model_state_dict.pth`.
    - Non-PyTorch models are saved as `<model_name}_model.joblib`.
    - Any exception during saving is caught and logged.
    """
    # Initialize lightweight logger
    logger = Logger(
        verbose=verbosity,
        log_dir=log_address,
        write_log=write_log,
    )
    # Validate and resolve output directory
    output_dir = validate_address(output_dir)
    try:
        # Save PyTorch models via state_dict
        if isinstance(model, nn.Module):
            address = output_dir / f"{model_name}_model_state_dict.pth"
            torch.save(model.state_dict(), address)
            logger(
                f"Saved {model_name} model state_dict to: {address}",
                verbosity=1,
            )
        # Save all other models via joblib
        else:
            model_path = output_dir / f"{model_name}_model.joblib"
            joblib.dump(  # pyright: ignore[reportUnknownMemberType]
                model,
                model_path,
            )
            logger(
                f"Saved {model_name} model to: {model_path}",
                verbosity=1,
            )
    except Exception as exc:
        # Catch and log any serialization errors
        logger(
            f"Error saving {model_name} model: {exc}",
            verbosity=0,
        )

def load_keras_model(
        model_path: Address,
        *,
        model_factory: Optional[Callable[[], Model]] = None,
        load_from_weights: bool = True,
    ) -> Model:
    """
    Load a Keras model from disk, preferring weight-based restoration
    with an automatic fallback to full model loading.

    This function attempts to restore model weights into a freshly
    constructed architecture when ``load_from_weights`` is True.
    If weight loading fails (e.g., incompatible file or missing
    weights), it falls back to loading a fully serialized model.

    Parameters
    ----------
    model_path : Address
        Path to the saved model weights or serialized model file.
    model_factory : Optional[Callable[[], keras.Model]]
        Factory function that returns an uninitialized model with the
        correct architecture. Required for weight-based loading.
    load_from_weights : bool, default=True
        Whether to attempt loading model weights first.

    Returns
    -------
    keras.Model
        Loaded Keras model ready for inference or training.

    Raises
    ------
    RuntimeError
        If the loaded object is not a Keras model.
    FileNotFoundError
        If ``model_path`` does not exist.
    """
    model_path = validate_address(model_path)
    # Preferred path: attempt to load weights into a fresh model
    if load_from_weights and model_factory is not None:
        try:
            model = model_factory()
            model.load_weights(  # pyright: ignore[reportUnknownMemberType]
                model_path
            )
            return model
        except Exception as exc:
            # Weight loading failed; fall back to full model loading
            print(
                f"[WARN] Failed to load weights from '{model_path}'. "
                f"Falling back to full model load. Reason: {exc}"
            )

    # Fallback path: load a fully serialized model
    loaded = joblib.load(  # pyright: ignore[reportUnknownMemberType]
        model_path
    )
    if not isinstance(loaded, Model):
        raise RuntimeError(
            f"Object loaded from '{model_path}' is not a Keras Model "
            f"(got {type(loaded).__name__})."
        )

    return loaded

