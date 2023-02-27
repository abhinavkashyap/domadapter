from domadapter.commands.download import download
from domadapter.commands.results import results
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
    domadapter_group.add_command(results)
    domadapter_group()


if __name__ == "__main__":
    main()
