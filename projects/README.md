# Projects

## Package Managers

1. Python Virtual Environments (vnev)
2. Poetry

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

5. install librarye

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
