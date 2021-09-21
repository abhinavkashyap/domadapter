import zipfile
from typing import Optional


def unzip_file(filepath: str, destination_dir: Optional[str] = None):
    """Unzips file to the given directory.

    Parameters
    ----------
    filepath: str
        The path of the zip file
    destination_dir: Optional[str]
        The destination directory
        If None then the current directory
        from which the command is executed is used.

    Returns
    -------

    """
    with zipfile.ZipFile(str(filepath), "r") as zip_ref:
        zip_ref.extractall(destination_dir)
