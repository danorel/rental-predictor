from pathlib import Path
from dotenv import load_dotenv


def load_environment(
        path,
        filename
):
    """
    Load the environment due to the mentioned state.
    State = {'dev', 'prod', 'test'}
    :type path: str
    :type filename: str
    """
    env_path = Path('.') / f'{path}' / f'{filename}'
    print(f"Loading environment variables from {env_path}")
    load_dotenv(
        dotenv_path=env_path,
        verbose=True
    )
