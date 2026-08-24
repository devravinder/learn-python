# Env Setup

## Local ENV Setup

### Installation

- install IDE: `VSC`
  - extensions:
    - python
    - autopep8 (for formatting)
    - Jupyter ( pack : install all official extensions  )

- install python & its related tools in system level
  - `pyenv`
  - `pyton3`  ( v3.14.7)
  - `python3.17-venv` - for virtual env
  - `pip`             - package manager
  - `uv`              - modern Python package/project manager

### Folder & Env Setup

- create work space folder : `python`
- create Virtual ENV : `python -m venv .venv`
  - run inside `python` workspace folder

- activate Virtual ENV : `source .venv/bin/activate`
  - run inside `python` workspace folder
  - General-purpose Python environment
    - later we can use project level venv with uv package manager

  - Now your terminal uses Python from venv instead of system Python
  - Packages you install will go only inside this project
  - check `which python`

  - Note:-
    - to deactive venv ( after this usage ): `deactivate`

### Installing Packages / libs

- `python -m pip install numpy`  
  - run inside `python` workspace folder
  - install pcakges within with workspace project

### Running Python Scripts

#### Simple Python Scrtipt (`.py`)

- `python hello.py`
- note:
  - the python script file should be inside `python` workspace folder ( or subfolder )
  - else we have to setup proper venv for that files / project

#### Shortcut Key setup : `Ctrl + R, Ctrl + P`

1. goto keyboard shortcuts (Ctrl + K, Ctrl + S)
2. click on json symbol (keybindings.json)
3. add the below code:

```json
{
  "key": "ctrl+r ctrl+p",
  "command": "workbench.action.terminal.sendSequence",
  "args": { "text": "python ${file}\u000D" }
}
```

`\u000D` means Enter key automatically.

#### Interactive Python Note Book (`.ipynb`)

- `.ipynb` requires
  - VS Code Jupyter extension
  - Python environment
  - `ipykernel` installed in that environment
    - install : `python -m pip install ipykernel`

- Configure Python Kernal in VSC
  - open any `.ipynb` ( create if needed )
  - on top right click kernal / Select Kernal
    - then `Select Another Kernal` > `Pyton Environment` > `Create Python Environment`
    - then give path to : `.venv/bin/python`  - which is inside `python` workspace folder
      - `/python/.venv/bin/python`

## Cloud ENV Setup

- [Google Colab](https://colab.research.google.com/)
- create & run `.ipynb` in browser directly
  - recomonded ( very easy setup  & free resource )

## Observations

- `python -m pip install numpy` vs `pip install numpy`
  - `pip install numpy`
    - uses the `pip` executable found in PATH / what shell finds first

  - `python -m pip install numpy`
    - uses pip associated with the selected Python interpreter
    - recomonded
