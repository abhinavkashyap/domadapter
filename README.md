# Project Title

`Domadapter` aims to train adapters for NLP domain adaptation. 

*This is a work in progress. Contact the authors if you face difficulties using it*


## Environment Variables (For Developers)

To run this project, you will need to install `direnv`. `direnv` exports 
environment variables to the shell. You can install `direnv` for your shell 
following the information at [https://direnv.net/](https://direnv.net/)

We use environment variables to store common paths to cachÃ© pretrained models, datasets etc. 
## Installation

You need `poetry` to install the project. 
Visit [https://python-poetry.org/](https://python-poetry.org/)
for installation instructions. 

**_Creating a virtualenv_**

- Use ``poetry shell`` to create a virtual environment 
- run ``poetry install`` to install the dependencies

Commit your `poetry.lock` file if you install any new library.
## Running Tests

We are using [https://ward.readthedocs.io](https://ward.readthedocs.io) to run our tests.
Run ``ward --path tests`` to run all the tests. Currently the tests are really slow. 
It takes a lot of time. 

  