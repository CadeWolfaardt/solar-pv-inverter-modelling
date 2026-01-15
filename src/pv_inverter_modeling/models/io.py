# stdlib
from typing import Any
from pathlib import Path
# thirdpartylib
import joblib # pyright: ignore[reportMissingTypeStubs]
import torch
import torch.nn as nn
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
