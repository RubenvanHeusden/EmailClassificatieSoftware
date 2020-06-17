"""
This file contains several utility functions used by the other scripts in this module
"""
import os
import torch
import tarfile
import pandas as pd
from tqdm import tqdm
from configurations import ROOT_DIR
from urllib.request import urlretrieve
from code_utils.dataiterator import DataIterator
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, recall_score, precision_score


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def generic_training(model, criterion, optimizer, scheduler, dataset, n_epochs=5, device=torch.device("cpu"),
                     save_path=None, save_name=None, tensorboard_dir=False, checkpoint_interval=5, clip_val=0):

    # Set the model in training mode
    model.to(device)
    model.train()
    if tensorboard_dir:
        writer = SummaryWriter(tensorboard_dir)

    for epoch in range(n_epochs):
        if save_path:
            if epoch % checkpoint_interval == 0:
                torch.save(model.state_dict(), "%s/%s_epoch_%d.pt" % (save_path, save_name, epoch))

        # Calculate several training statistics
        all_predictions = []
        all_ground_truth_labels = []
        epoch_running_loss = 0.0

        for i, batch in tqdm(enumerate(dataset)):
            optimizer.zero_grad()
            X, y = batch
            y = y.to(device)
            # Whether the padding should be removed when fed into the LSTM
            if isinstance(X, tuple):
                X = list(X)
                for z in range(len(X)):
                    X[z] = X[z].to(device)
            else:
                X = X.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            # training the network
            loss.backward()
            if clip_val:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            optimizer.step()

            all_predictions.extend(outputs.detach().cpu().argmax(1).tolist())
            all_ground_truth_labels.extend(y.cpu().tolist())
            epoch_running_loss += loss.item()

        scheduler.step()
        correct_list = [1 if a == b else 0 for a, b in zip(all_predictions, all_ground_truth_labels)]
        acc = sum(correct_list) / len(correct_list)
        if tensorboard_dir:
            writer.add_scalar('loss', epoch_running_loss, epoch)
            writer.add_scalar('accuracy', acc, epoch)
    if tensorboard_dir:
        writer.close()
    return model


def generic_evaluation(model, dataset, criterion, device=None):
    model.to(device)
    # Set the model to evaluation mode, important because of the Dropout Layers
    model = model.eval()
    # Calculate several test statistics
    epoch_running_loss = 0.0
    all_predictions = []
    all_ground_truth_labels = []
    all_texts = []
    for i, batch in tqdm(enumerate(dataset)):
        X, y = batch
        y = y.to(device)
        if isinstance(X, tuple):
            X = list(X)
            for z in range(len(X)):
                X[z] = X[z].to(device)
        else:
            X = X.to(device)
        outputs = model(X)

        loss = criterion(outputs, y)
        epoch_running_loss += loss.item()
        # Calculate several batch statistics
        all_predictions.extend(outputs.detach().cpu().argmax(1).tolist())
        all_ground_truth_labels.extend(y.cpu().tolist())
        all_texts.append(X)
    correct_list = [1 if a == b else 0 for a, b in zip(all_predictions, all_ground_truth_labels)]
    acc = sum(correct_list) / len(correct_list)
    prog_string = "[|Train| Loss: %.3f, Acc: %.3f, f_1: %.3f, recall: %.3f, precision, %.3f]" \
                  % (epoch_running_loss, acc,
                     f1_score(all_ground_truth_labels, all_predictions, average="weighted"),
                     recall_score(all_ground_truth_labels, all_predictions, average="weighted"),
                     precision_score(all_ground_truth_labels, all_predictions, average="weighted"))

    print(prog_string)
    return all_predictions, all_ground_truth_labels, all_texts


def single_task_class_weighting(dataset: DataIterator) -> torch.Tensor:
    """

    :param dataset: A dataset contained in the DataIterator class for which class weights
    should be calculated.

    :return: torch.Tensor of size [num_unique_classes, 1] specifying the class weight for each one of them
    """
    total_y = []
    for X, y in dataset:
        total_y.append(y)

    total_y = torch.cat(total_y, dim=0)
    classes = torch.unique(total_y)
    weights = compute_class_weight('balanced', classes=classes.data.numpy(),
                                   y=total_y.data.numpy())
    return torch.from_numpy(weights).float()


def download_word_embeddings_nl() -> None:
    """
    Method that is used to download the NIPS Dutch word embeddings
    and put them in the 'resources/word_embeddings' folder. Also includes
    code to show the progress of the download as this can take a few
    minutes based on the speed of the internet connection

    :return: None
    """
    print('--- Beginning word embedding file download ---')
    url = 'https://www.clips.uantwerpen.be/dutchembeddings/combined-320.tar.gz'
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                  desc=url.split('/')[-1]) as t:
        file_tmp = urlretrieve(url, filename=None, reporthook=t.update_to)[0]
        t.total = t.n

    base_name = os.path.basename(url)
    file_name, file_extension = os.path.splitext(base_name)
    tar = tarfile.open(file_tmp)
    tar.extractall(ROOT_DIR+'/resources/word_embeddings/'+file_name)
    return None


def embeddings_available() -> bool:
    """
    Method that checks whether there exists a file with the name of the
    word embeddings file in the current package folder. This is used to
    check whether the embeddings should be downloaded or not.

    :return:
    """
    all_files = []
    for root, dirs, files in os.walk(ROOT_DIR):
        all_files.extend(files)
    if 'combined-320.txt' not in all_files:
        return False
    else:
        return True


def get_num_labels_from_file(file_name: str, delimiter: str = ",", quotechar: str = '"',
                             label_col_name: str = 'label') -> int:
    """
    Method that can be used to retrieve the unique numbers of labels in a file.

    :param file_name: string specifying the name and location of the file where the data is located.
    :param delimiter: string specifying the delimiter used in the csv file
    :param quotechar: string specifying the quotechar used in the csv file
    :param label_col_name: string specifying the name of the header of the column where the labels are located.
    :return: Returns the (integer) amount of unique labels in the dataset
    """
    return pd.read_csv(file_name, sep=delimiter, quotechar=quotechar)[label_col_name].nunique()
