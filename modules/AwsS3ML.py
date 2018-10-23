from AwsS3 import AwsS3
from CVFileMixer import CVFileMixer


class AwsS3ML:
    def __init__(self, aws_profile_name=None):
        self.S3 = AwsS3(aws_profile_name)

    def read_cv_parquets(self, parquet_name, n_splits, validation_index, s3_bucket, s3_path, delete_local=True):
        if '.parquet' in parquet_name: parquet_name = parquet_name.replace('.parquet', '')

        file_names = ['{}_{}.parquet'.format(parquet_name, i) for i in range(n_splits)]
        df_list = [self.S3.read_parquet(name, s3_bucket, s3_path, delete_local) for name in file_names]

        return CVFileMixer.get(df_list, validation_index)


if __name__ == '__main__':
    s3 = AwsS3ML()
    s3.read_cv_parquets('abcd.parquet', 10, '', '')

