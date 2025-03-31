import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

project_name="ML_codes" 

list_of_files=[
    ".github/workflows/gitkeep",
    f"Source/{project_name}/__init__.py",
    f"Source/{project_name}/components/__init__.py",
    f"Source/{project_name}/components/data_ingestion.py",
    f"Source/{project_name}/components/data_transformation.py",
    f"Source/{project_name}/components/model_trainer.py",
    f"Source/{project_name}/components/model_monitoring.py",
    f"Source/{project_name}/pipelines/__init__.py",
    f"Source/{project_name}/pipelines/training_pipeline.py",
    f"Source/{project_name}/exception.py",
    f"Source/{project_name}/logger.py",
    f"Source/{project_name}/utils.py",
    "app.py",
    "setup.py",
    "Dockerfile",
    "requirement.txt",
    "Notebook/exploratory_data_analysis.ipynb",
    "Notebook/Model_Training.ipynb"
]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir, filename= os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,"w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")