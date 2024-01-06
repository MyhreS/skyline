
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()

class BinaryDroneClassificator:
    def __init__(self):
        self.data_path = os.getenv("BINARY_DRONE_DATA_PATH")

    def run(self):
        logging.info("Running binary drone classificator")
        print("Data path: {}".format(self.data_path))
        print(os.listdir(self.data_path))


if __name__ == "__main__":
    binary_drone_classificator = BinaryDroneClassificator()
    binary_drone_classificator.run()
