import os
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml
from dotenv import load_dotenv

load_dotenv()

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_names = self.config['bucket_file_names']

        os.makedirs(RAW_DIR, exist_ok = True)
        logger.info("Data Ingestion Started.")

    def download_csv_fron_gcp(self):
        try:
            client = storage.Client(project = GOOGLE_CLOUD_PROJECT)
            bucket = client.bucket(self.bucket_name)
            for file_name in self.file_names:
                file_path = os.path.join(RAW_DIR, file_name)

                if file_name == 'animelist.csv':
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)

                    data = pd.read_csv(file_path, nrows = 5000000)
                    data.to_csv(file_path, index = False)
                    logger.info("Large file detected only downloading 5M rows ")

                else :
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)

                    logger.info("Downloading Smaller Files anime and anime_with_synopsis")

        except Exception as e:
            logger.error("Error while downloading data from GCP.")
            raise CustomException("Failed to download data.",e)
        
    def run(self):
        try:
            logger.info("Starting data ingestion process...")
            self.download_csv_fron_gcp()
            logger.info("Data ingestion completed.")
        except CustomException as ce:
            logger.error(f"CustomException : {str(ce)}")
        finally:
            logger.info("Data Ingestion Done.")

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()