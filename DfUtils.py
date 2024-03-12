import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

def read_parquet_in_batches(file_path, selections=None, batch_size=1000000):
    """
    Read a Parquet file in batches and return a concatenated DataFrame.

    Parameters:
    file_path (str): The path to the Parquet file.
    selections (str, optional): A string representing the selection criteria to apply to each batch. Defaults to None.
    batch_size (int, optional): The number of rows to read per batch. Defaults to 1000000.

    Returns:
    pandas.DataFrame: The concatenated DataFrame.

    """
    parquet_file = pq.ParquetFile(file_path)
    df = []
    for batch in parquet_file.iter_batches(batch_size):
        batch_df = batch.to_pandas()
        if selections is not None:
            batch_df = batch_df.query(selections)
        df.append(batch_df)
    return pd.concat(df)

def apply_model_in_batches(ModelHandl, ColsToKeep, file_path, selections=None, batch_size=1000000):
    """
    Applies a machine learning model to a large dataset in batches.

    Args:
        ModelHandl: A hipe4ml ModelHandl.
        ColsToKeep: A list of column names to keep in the resulting DataFrame.
        file_path: The path to the Parquet file containing the dataset.
        selections: An optional string representing the query to apply to the dataset.
        batch_size: The size of each batch to process.

    Returns:
        A pandas DataFrame containing the results of applying the model to the dataset.
    """
    parquet_file = pq.ParquetFile(file_path)
    df = []
    for batch in parquet_file.iter_batches(batch_size):
        batch_df = batch.to_pandas()
        if selections:
            batch_df = batch_df.query(selections)
        pred = ModelHandl.predict(batch_df, False)
        batch_df = batch_df.loc[:, ColsToKeep]
        batch_df['ML_output_Bkg'] = pred[:, 0]
        batch_df['ML_output_Prompt'] = pred[:, 1]
        batch_df['ML_output_FD'] = pred[:, 2]
        df.append(batch_df)
    return pd.concat(df)

def split_and_dump_parquet(df, output_filename, suffixes, train_frac=0.8):
    """
    Split the input DataFrame into train and test sets, and save them as Parquet files.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame to be split.
    - output_filename (str): The base path for the output Parquet files.
    - suffixes (list): A list of suffixes to be appended to the output filenames.
    - train_frac (float): The fraction of data to be used for training. Default is 0.8.

    Returns:
    None
    """
    df_train, df_test = train_test_split(df, train_size=train_frac, random_state=42)
    df_train.to_parquet(output_filename + suffixes[0], engine='fastparquet')
    df_test.to_parquet(output_filename + suffixes[1], engine='fastparquet')
    del df, df_train, df_test

