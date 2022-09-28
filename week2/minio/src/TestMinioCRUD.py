from minioCRUD import MinioCRUD
from pathlib import Path
import pytest
import time


@pytest.fixture
def minioCRUD() -> MinioCRUD:
    return MinioCRUD("localhost:9000", "minio", 'minio123')

@pytest.fixture
def bucket_name() -> str:
    return 'bucket1'


def test_buckets(minioCRUD, bucket_name):
    minioCRUD.make_bucket(bucket_name)
    time.sleep(1)
    buckets_list = minioCRUD.get_buckets_list()
    assert bucket_name in [b.name for b in buckets_list]

    minioCRUD.remove_bucket(bucket_name)
    time.sleep(1)
    buckets_list = minioCRUD.get_buckets_list()
    assert bucket_name not in [b.name for b in buckets_list]


def test_files(minioCRUD, bucket_name):
    p = Path("../")
    files_list = [a for a in p.glob('*.*')]
    assert len(files_list) != 0

    minioCRUD.make_bucket(bucket_name)
    time.sleep(1)

    print(files_list)
    for fname in files_list:
        minioCRUD.upload_file(bucket_name, fname)
    time.sleep(1)

    remote_file_list = [obj._object_name for obj in minioCRUD.get_list_objects(bucket_name)]
    files_list = [str(fp.name) for fp in files_list]
    assert set(files_list).issubset(set(remote_file_list))

    for fname in files_list:
        minioCRUD.remove_object(bucket_name, fname)
    time.sleep(1)
    assert len(list(minioCRUD.get_list_objects(bucket_name))) == 0

    minioCRUD.remove_bucket(bucket_name)