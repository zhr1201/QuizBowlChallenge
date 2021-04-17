import click
import subprocess
from os import path, makedirs, remove
import zipfile


DS_VERSION = '2018.04.18'
S3_HTTP_PREFIX = 'https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/'
QANTA_MAPPED_DATASET_PATH = f'qanta.mapped.{DS_VERSION}.json'
QANTA_TRAIN_DATASET_PATH = f'qanta.train.{DS_VERSION}.json'
QANTA_DEV_DATASET_PATH = f'qanta.dev.{DS_VERSION}.json'
QANTA_TEST_DATASET_PATH = f'qanta.test.{DS_VERSION}.json'

FILES = [
	QANTA_MAPPED_DATASET_PATH,
	QANTA_TRAIN_DATASET_PATH,
	QANTA_DEV_DATASET_PATH,
	QANTA_TEST_DATASET_PATH
]

# OBJSTORE_PREFIX = 'https://obj.umiacs.umd.edu/processed_tossup/'
# QANTA_TRAIN_RETRIEVED_PARAGRAPHS = f'qanta.train.paragraphs.{DS_VERSION}.jsonl.zip'
# QANTA_DEV_RETRIEVED_PARAGRAPHS = f'qanta.dev.paragraphs.{DS_VERSION}.json'
# QANTA_TEST_RETRIEVED_PARAGRAPHS = f'qanta.test.paragraphs.{DS_VERSION}.json'
OBJSTORE_PREFIX = 'https://s3-us-west-2.amazonaws.com/pinafore-us-west-2/qanta-jmlr-datasets/rc/'
QANTA_TRAIN_RETRIEVED_SENTS = f'evidence_docs_train.json'
QANTA_TRAIN_RETRIEVED_SENTS_WITH_TEXT = f'evidence_docs_train_with_sent_text.json'
QANTA_DEV_RETRIEVED_SENTS = f'evidence_docs_dev.json'
QANTA_DEV_RETRIEVED_SENTS_WITH_TEXT = f'evidence_docs_dev_with_sent_text.json'
QANTA_TEST_RETRIEVED_SENTS = f'evidence_docs_test.json'
QANTA_TEST_RETRIEVED_SENTS_WITH_TEXT = f'evidence_docs_test_with_sent_text.json'

# PARAGRAPH_FILES = [
#     QANTA_TRAIN_RETRIEVED_PARAGRAPHS,
#     QANTA_DEV_RETRIEVED_PARAGRAPHS,
#     QANTA_TEST_RETRIEVED_PARAGRAPHS
# ]
PARAGRAPH_FILES = [
	QANTA_TRAIN_RETRIEVED_SENTS,
	QANTA_DEV_RETRIEVED_SENTS,
	QANTA_TEST_RETRIEVED_SENTS,
	QANTA_TRAIN_RETRIEVED_SENTS_WITH_TEXT,
	QANTA_DEV_RETRIEVED_SENTS_WITH_TEXT,
	QANTA_TEST_RETRIEVED_SENTS_WITH_TEXT
]


def make_file_pairs(file_list, source_prefix, target_prefix):
	return [(path.join(source_prefix, f), path.join(target_prefix, f)) for f in file_list]


def shell(command):
	return subprocess.run(command, check=True, shell=True, stderr=subprocess.STDOUT)


def download_file(http_location, local_location):
	print(f'Downloading {http_location} to {local_location}')
	makedirs(path.dirname(local_location), exist_ok=True)
	print(f'wget -O {local_location} {http_location}')
	# shell(f'wget -O {local_location} {http_location}')


def download(local_qanta_prefix, retrieve_paragraphs=False):
	"""
	Download the qanta dataset
	"""
	for s3_file, local_file in make_file_pairs(FILES, S3_HTTP_PREFIX, local_qanta_prefix):
		download_file(s3_file, local_file)

	if retrieve_paragraphs:
		for objstore_file, local_file in make_file_pairs(PARAGRAPH_FILES, OBJSTORE_PREFIX, local_qanta_prefix):
			download_file(objstore_file, local_file)
#             with zipfile.ZipFile(local_file, 'r') as zip_file:    
#                 zip_file.extractall(local_qanta_prefix)
#                 remove(local_file)
