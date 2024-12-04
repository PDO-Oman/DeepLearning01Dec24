from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate with Kaggle
api = KaggleApi()
api.authenticate()

# Specify the dataset name (e.g., Titanic dataset)
dataset_name = 'titanic'

# Download and unzip dataset to the current directory
api.dataset_download_files(dataset_name, path='./', unzip=True)
