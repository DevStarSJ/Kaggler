import os
import boto3
import pickle
import pandas as pd

class AwsS3:
    def __init__(self, aws_profile_name=None):
        self.s3_session = boto3.Session(profile_name=aws_profile_name)
        self.s3_client = self.s3_session.client('s3')

    def to_parquet(self, df, parquet_name, s3_bucket, s3_path, delete_local=True):
        local_url = './{}'.format(parquet_name)

        df.to_parquet(local_url, engine='pyarrow', compression='gzip')
        self.to_s3(parquet_name, s3_bucket, s3_path)
        if delete_local: os.remove(local_url)

    def read_parquet(self, parquet_name, s3_bucket, s3_path, delete_local=True):
        local_url = './{}'.format(parquet_name)

        self.read_s3(parquet_name, s3_bucket, s3_path)
        df = pd.read_parquet(local_url)
        if delete_local: os.remove(local_url)

        return df

    def to_pickle(self, obj, pickle_name, s3_bucket, s3_path, delete_local=True):
        local_url = './{}'.format(pickle_name)

        with open(local_url, 'wb') as f:
            pickle.dump(obj, f, protocol=4)

        self.to_s3(pickle_name, s3_bucket, s3_path)
        if delete_local: os.remove(local_url)

    def read_pickle(self, pickle_name, s3_bucket, s3_path, delete_local=True):
        local_url = './{}'.format(pickle_name)

        self.read_s3(pickle_name, s3_bucket, s3_path)

        obj = None
        with open(local_url, 'rb') as f:
            obj = pickle.load(f)

        if delete_local: os.remove(local_url)

        return obj

    def to_csv(self, df, name, s3_bucket, s3_path, header=False, delete_local=True):
        local_url = './{}'.format(name)

        df.to_csv(local_url, header=header, index=False)
        self.to_s3(name, s3_bucket, s3_path)
        if delete_local: os.remove(local_url)

    def to_s3(self, file_name, s3_bucket, s3_path):
        local_url = './{}'.format(file_name)
        s3_url = '{}{}'.format(s3_path, file_name)
        with open(local_url, 'rb') as data:
            self.s3_client.upload_fileobj(data, s3_bucket, s3_url)

    def read_s3(self, file_name, s3_bucket, s3_path):
        local_url = './{}'.format(file_name)
        s3_url = '{}{}'.format(s3_path, file_name)
        self.s3_client.download_file(s3_bucket, s3_url, local_url)