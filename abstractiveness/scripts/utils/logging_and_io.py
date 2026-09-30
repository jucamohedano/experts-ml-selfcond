"""
Logging to the run folder and CSV writing.

Generic helpers used by the executor and every module.

Explanations: none needed.
"""

import logging
import pathlib
import pandas as pd


formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def set_folder_log(folder_path: pathlib.Path) -> None:
    """Configure logging to write output to a main.log file in the specified folder."""
    folder_path.mkdir(parents=True, exist_ok=True)

    log_file_path = folder_path / "main.log"
    root_logger = logging.getLogger()

    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def save_dataframe(output_dataframe: pd.DataFrame, out_path: pathlib.Path, index: bool = False) -> pathlib.Path:
    """Save a pandas DataFrame to a CSV file at the specified output path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_dataframe.to_csv(out_path, index=index)
    return out_path
