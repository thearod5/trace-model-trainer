import os

from dotenv import load_dotenv

load_dotenv()

OUTPUT_PATH = os.path.expanduser(os.environ["OUTPUT_PATH"])
