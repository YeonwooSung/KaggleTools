import pandas as pd
import numpy as np


def dropnan_with_two_ways(df: pd.DataFrame, use_memory_efficient:bool=True) -> pd.DataFrame:
    '''
    Try to use "inplace=True" rather than assigning to a new variable.
    By reassigning to a new variable, you are creating a new object in memory, 
    which means that the old object is still in memory and you are using more memory than you need to.
    '''
    if use_memory_efficient:
        df.dropna(inplace=True)
    else:
        df = df.dropna()
    return df


def read_only_required_columns(file_path:str):
    '''
    Read only required columns from a csv file.
    '''
    df = pd.read_csv(file_path, usecols=['col1', 'col2', 'col3'])
    return df


def read_only_required_rows(file_path:str):
    '''
    Read only required rows from a csv file.
    '''
    df = pd.read_csv(file_path, skiprows=range(1, 100))
    return df


def visualize_memory_usage(df: pd.DataFrame):
    '''
    Visualize memory usage of a dataframe.
    '''
    print(df.info(memory_usage='deep'))


def specify_column_data_type(
    filepath:str='dummy_dataset.csv', 
    col_list:list=["Employee_ID", "First_Name", "Salary", "Rating", "Country_Code"]
):
    '''
    Specify column data type.

    If you know the data type of a column and the max/min/average/median or distribution of the values in that column, 
    you can specify the data type of that column to be something more memory efficient.

    For exmaple, if you know that the values in a column are integers,
    you can specify the data type of that column to be an integer rather than a float.
    This will save memory because integers take up less memory than floats.
    '''
    df = pd.read_csv(
        "dummy_dataset.csv", 
        usecols=col_list, 
        dtype = {"Employee_ID":np.int32, "Country_Code":"category"}
    )
    return df


def convert_column_data_type(df: pd.DataFrame):
    '''
    Convert column data type.

    If you know the data type of a column and the max/min/average/median or distribution of the values in that column, 
    you can convert the data type of that column to be something more memory efficient.

    For exmaple, if you know that the values in a column are integers,
    you can convert the data type of that column to be an integer rather than a float.
    This will save memory because integers take up less memory than floats.
    '''
    df["Employee_ID"] = df["Employee_ID"].astype(np.int32)
    df["Country_Code"] = df["Country_Code"].astype("category")
    return df


def read_data_as_chunk(file_path:str, chunk_size:int=1000):
    '''
    Read data as chunk.
    '''
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        print(chunk.shape)
