# Multiple Python Versions

## pyenv

### Setup

- install pyenv : `curl -fsSL https://pyenv.run | bash`
- then add to path ( or in `etc/bash.bashrc` )

  ```shell
    export PYENV_ROOT=/home/ravinder/.pyenv
    export PATH=$PYENV_ROOT/bin:$PATH
  ```

  - here: `/home/ravinder/.pyenv` is installed folder

- intialize:
  - `pyenv init --install`
  - `pyenv init bash`

### Usage

- to list installed version: `pyenv versions`
- to see avialable versions: `pyenv install -l`
- install a version: `pyenv install 3.14.7`
- set globally: `pyenv global 3.14.7`
- set locally with in a project: `pyenv local 3.14.7`
- to delete a version: `pyenv uninstall 3.14.4`
- update pyenv: `pyenv update`
