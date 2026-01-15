# stdlib
from pathlib import Path
from datetime import datetime
from typing import Optional
from types import TracebackType
# projectlib
from pv_inverter_modeling.utils.paths import validate_address
from pv_inverter_modeling.utils.typing import Verbosity, Address

class Logger(object):
    """
    Lightweight callable logger with optional file persistence.

    This class provides a minimal logging utility that supports
    verbosity-based message filtering and can either print messages
    to stdout or append them to a log file. It is intended for simple
    pipeline and script-level logging where a full logging framework
    would be excessive.
    """

    def __init__(
        self,
        verbose: Verbosity = 0,
        log_dir: Address = Path.cwd(),
        write_log: bool = False
    ) -> None:
        """
        Initialize the logger.

        Parameters
        ----------
        verbose : Verbosity, default 0
            Verbosity threshold. Messages with a verbosity level less
            than or equal to this value will be emitted.
        log_dir : Address, default Path.cwd()
            Directory in which the log file will be written if
            `write_log` is True. The file name is fixed as `log.txt`.
        write_log : bool, default False
            If True, messages are appended to a log file. If False,
            messages are printed to stdout.
        """
        self.verbose = verbose
        # Resolve and validate output path for the log file
        self.log_path = validate_address(log_dir) / "log.txt"
        # Toggle between stdout printing and file logging
        self.write_log = write_log

    def __call__(self, msg: str, verbosity: int = 0) -> None:
        """
        Emit a log message if the verbosity threshold is met.

        This allows the logger instance to be used as a callable,
        e.g. `logger("message", verbosity=1)`.

        Parameters
        ----------
        msg : str
            Message to be logged.
        verbosity : int, default 0
            Verbosity level associated with the message. The message
            is emitted only if `self.verbose >= verbosity`.
        """
        if self.verbose >= verbosity:
            formatted = self._format(msg)
            if self.write_log:
                self.write(formatted)
            else:
                print(formatted)

    def write(self, msg: str) -> None:
        """
        Append a formatted message to the log file.

        Parameters
        ----------
        msg : str
            Message to append to the log file.
        """
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(msg + "\n")
    
    def _format(self, msg: str) -> str:
        """
        Format a log message with a timestamp.

        Parameters
        ----------
        msg : str
            Raw log message.

        Returns
        -------
        str
            Timestamp-prefixed log message.
        """
        ts = datetime.now().isoformat(timespec="seconds")
        return f"[{ts}] {msg}"
    
    def __enter__(self) -> "Logger":
        return self

    def __exit__(
            self, 
            exc_type: Optional[type[BaseException]],
            exc: Optional[BaseException],
            tb: Optional[TracebackType]
        ) -> None:
        pass