"""
This file contains several utility functions used by the other scripts in this module
"""

from urllib.request import urlretrieve
import os
import tarfile
import pathlib
import glob
from configurations import ROOT_DIR


def generic_training():
    pass


def generic_evaluation():
    pass


def download_word_embeddings_nl(path='resources/word_embeddings/'):
    print('Beginning file download with urllib2...')
    url = 'https://www.clips.uantwerpen.be/dutchembeddings/combined-160.tar.gz'
    file_tmp = urlretrieve(url, filename=None)[0]
    base_name = os.path.basename(url)
    file_name, file_extension = os.path.splitext(base_name)
    tar = tarfile.open(file_tmp)
    tar.extractall('resources/word_embeddings/'+file_name)


def embeddings_available():
    directories = []
    for root, dirs, files in os.walk(ROOT_DIR):
        directories.extend(dirs)
    if 'word_embeddings' not in directories:
        return False
    else:
        return True





