import subprocess



def is_command_available(command: str):
    try:
        result = subprocess.run([command, '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False