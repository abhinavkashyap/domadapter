from domadapter.commands.download import download
import click
from art import tprint


@click.group(name="domadapter")
def domadapter_group():
    """Root Command"""
    pass


def main():
    # Prints the ascii art
    tprint("Dom Adapters")
    domadapter_group.add_command(download)
    domadapter_group()


if __name__ == "__main__":
    main()
