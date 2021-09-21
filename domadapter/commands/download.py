import click
from datasets import load_dataset
from domadapter.utils.download_utils import GoogleDriveDownloader
from domadapter.utils.general_utils import unzip_file
from domadapter.utils.mnli_utils import prepare_mnli
from domadapter.console import console
import os
from pathlib import Path


@click.group()
def download():
    pass


@download.command()
def sa():
    """Download Sentiment Analysis Dataset for Unsupervised Domain Adaptation"""
    downloader = GoogleDriveDownloader()
    dataset_cache = os.environ["DATASET_CACHE_DIR"]
    dataset_cache = Path(dataset_cache)
    destination_file = dataset_cache.joinpath("sa.zip")

    with console.status("Downloading SA data"):
        downloader.download_file_from_google_drive(
            file_id="1mQICb7-mpJC2tR3xP2SsNkvihVMwKCjw", destination=destination_file
        )

    console.print(f"[green] Downloaded SA Data")

    # unzip sst
    with console.status("Extracting SA"):
        unzip_file(filepath=str(destination_file), destination_dir=str(dataset_cache))

    console.print(f"[green] Extracted SA Data")
    # delete the zip file
    destination_file.unlink()

    console.print(f"[green] SA Data Available")


@download.command()
def mnli():
    """Download Multi-Genre Natural Language Inference Dataset for Unsupervised Domain Adaptation"""

    load_dataset("multi_nli")

    prepare_mnli()

    console.print(f"[green] MNLI Data Available")


if __name__ == "__main__":
    download()
