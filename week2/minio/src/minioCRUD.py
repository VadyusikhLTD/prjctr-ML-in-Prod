from minio import Minio
from pathlib import Path


class MinioCRUD:
    def __init__(self, endpoint: str, access_key: str, secret_key: str) -> None:
        self.client = Minio(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=False)

    # create / update
    def make_bucket(self, bucket_name):
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name=bucket_name)

    def upload_file(self, bucket_name: str, file_path: Path):
        self.client.fput_object(bucket_name=bucket_name, object_name=file_path.name, file_path=file_path)

    # read
    def get_buckets_list(self):
        return self.client.list_buckets()

    def get_list_objects(self, bucket_name):
        return self.client.list_objects(bucket_name)

    def download_file(self, bucket_name: str, object_name: str, file_path: Path):
        self.client.fget_object(bucket_name=bucket_name, object_name=object_name, file_path=str(file_path))

    # delete
    def remove_bucket(self, bucket_name):
        if self.client.bucket_exists(bucket_name):
            self.client.remove_bucket(bucket_name)

    def remove_object(self, bucket_name: str, object_name: str):
        self.client.remove_object(bucket_name=bucket_name, object_name=object_name)