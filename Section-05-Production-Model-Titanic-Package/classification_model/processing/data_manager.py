import typing as t

from pathlib import Path
import numpy as np
import re

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from classification_model import __version__ as _version
from classification_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

def remove_unwarranted_symbols(*, dataframe:pd.DataFrame) -> pd.DataFrame:

    dataframe = dataframe.copy()
    dataframe = dataframe.replace('?', np.nan)

    return (dataframe)


def get_first_cabin (*, row:str) -> str:
    
    try:
        return row.split()[0]
    except:
        return np.nan
    

def get_title (*, passenger:str) -> str:

    if re.search('Mrs', passenger):
        return 'Mrs'
    elif re.search('Mr', passenger):
        return 'Mr'
    elif re.search('Miss', passenger):
        return 'Miss'
    elif re.search('Master', passenger):
        return 'Master'
    else:
        return 'Other'


def pre_preocessing_pipeline(*, dataframe) -> pd.DataFrame:

    ########################################################################
    # Pre-Processing Steps
    #   1) Getthing the first Cabin among all the given cabins.
    #   2) Getting the title from the name of the passensger. 
    #   3) Removing the unwamnted the symbols from the dataFrame.
    #   4) Converting the Value type from string to float in Fare and Age.
    #   5) Dropping the unwanted columns in the dataframe.
    ########################################################################

    print("Type of Object", type(dataframe))
    print("dataframe", dataframe)

    dataframe['cabin'] = dataframe['cabin'].apply(lambda row: get_first_cabin(row=row))
    dataframe['name'] = dataframe['name'].apply(lambda passenger: get_title(passenger=passenger))
    dataframe = remove_unwarranted_symbols(dataframe=dataframe)
    dataframe['fare'] = dataframe['fare'].astype('float')
    dataframe['age'] = dataframe['age'].astype('float')
    dataframe.drop(labels=config.model_config.unused_vars, axis=1, inplace=True)


    return dataframe


def load_raw_dataset(*, file_name: str) -> pd.DataFrame:

    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe


def load_dataset(*, file_name: str) -> pd.DataFrame:
    
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    dataframe = pre_preocessing_pipeline(dataframe=dataframe)
    
    return dataframe


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
