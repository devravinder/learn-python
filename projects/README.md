# Projects

## Package Managers

1. Python Virtual Environments (vnev)
2. Poetry
3. UV (`pipx install uv`) ***

## Project with vnev

1. install supporting python package if needed for linux

   ```bash
     apt install python3.12-venv
   ```

2. create project dir

   ```bash
     mkdir calculator && cd calculator
   ```

3. create virtula env

   ```bash
     python3 -m venv venv
     # use module vnev & create folder vnev for package to staore
   ```

4. activate the virtual environment

   ```bash
     source venv/bin/activate
   ```

   - Now your terminal uses Python from venv instead of system Python
   - Packages you install will go only inside this project
   - check `which python3`

5. install library

   ```bash
   pip install colorama

   ```

6. Freeze dependencies (optional but recommended)

   ```bash
   pip freeze > requirements.txt

   ```

   later others can reinstall using:

   ```bash
   pip install -r requirements.txt
   ```

7. add the main code inside the calculator folder, create main.py
8. run

   ```bash
    python3 main.py

   ```

9. Deactivate virtual environment when done

   ```bash
     deactivate
   ```

## Project with Poetry

1. Install Poetry (if not installed)

   ```bash
      pipx install poetry
   ```

2. Create project directory using Poetry

   ```bash
      poetry new api-client
   ```

   - Creates folder `api-client`
   - Generates basic project structure + `pyproject.toml` (like package.json)

3. Move into project directory

   ```bash
   cd api-client
   ```

4. Install required library

   ```bash
   poetry add requests
   ```

   - Installs the `requests` package only for this project
   - Updates `pyproject.toml` and creates `poetry.lock`

5. Activate Poetry virtual environment

   ```bash
   poetry shell
   ```

   - Now your terminal uses the virtual environment created by Poetry
   - Check `which python3` to confirm

6. Add main code — inside `src/api_client/` folder create `main.py`
  **Note**:- **init**.py is to make our project as package

7. Run the project

   ```bash
   poetry shell
   python3 src/api_client/main.py # or
   poetry run python src/api_client/main.py
   ```

8. Deactivate / Exit Poetry shell when done

   ```bash
    exit
   ```

9. To install dependencies on another machine:
   1. cmd

      ```bash
         poetry install
      ```

   2. To run without entering shell:

      ```bash
         poetry run python api_cient/main.py
      ```

## Project With UV

1. Init project
   - `uv init uv-api-client`
2. add depedency
   - `uv add requests`
   - `uv add "flask=2.*'`
3. running
   - `uv run main.py`

4. to remove lib
   - `uv remove pandas`

5. Lock libraries

- `uv lock`

### Library with UV

1. `uv init --lib my-lib`
2. `uv publish my-lib`

### Mono-repo

1. Sub proejct
   - create a subproject in another project
   - `uv init my-child-app-1`

### Other Useful CMD

1. to list all pythons
   - `uv python list`
   - `uv python list --only-installed`

2. installing python version
   - `uv python install 3.13`

3. to change python version
   - `uv python pin 3.13` # run this in project folder

4. run python with specific version
   - `uv run python` # it uses active version from the project else global

5. Using tool

- `uv tool run jupyter lab` or `uvx jupyter lab`
