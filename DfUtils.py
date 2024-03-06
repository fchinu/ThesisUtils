import pandas as pd
import pyarrow.parquet as pq

def read_parquet_in_batches(file_path, selections = None, batch_size = 1000000):
    parquet_file = pq.ParquetFile(file_path)
    df = []
    for batch in parquet_file.iter_batches(batch_size):
        batch_df = batch.to_pandas()
        if selections is not None:
            batch_df = batch_df.query(selections)
        df.append(batch_df)
    return pd.concat(df)

def apply_model_in_batches(ModelHandl, ColsToKeep, file_path, selections = None, batch_size = 1000000):
    parquet_file = pq.ParquetFile(file_path)
    df = []
    for batch in parquet_file.iter_batches(batch_size):
        batch_df = batch.to_pandas()
        batch_df = batch_df.query(selections)
        pred = ModelHandl.predict(batch_df, False)
        batch_df = batch_df.loc[:, ColsToKeep]
        batch_df['ML_output_bkg'] = pred[:, 0]
        batch_df['ML_output_prompt'] = pred[:, 1]
        batch_df['ML_output_FD'] = pred[:, 2]
        df.append(batch_df)
    return pd.concat(df)