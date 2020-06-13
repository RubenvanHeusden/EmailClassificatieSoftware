"""
This file contains several utility functions used by the other scripts in this module
"""


import os
import torch
import tarfile
from tqdm import tqdm
from configurations import ROOT_DIR
from urllib.request import urlretrieve
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, recall_score, precision_score


def train(model, criterion, optimizer, scheduler, dataset, n_epochs=5, device=torch.device("cpu"), include_lengths=False,
           save_path=None, save_name=None, tensorboard_dir=False, checkpoint_interval=5, clip_val=0):
    # Set the model in training mode just to be safe
    model.to(device)
    model.train()
    if tensorboard_dir:
        writer = SummaryWriter(tensorboard_dir)
        # if include_lengths:
        #     s = dataset.sample().cuda()
        #     sample = s, torch.tensor([s.shape[0]]).cuda()
        # else:
        #     sample = dataset.sample().cuda().unsqueeze(0)
        # writer.add_graph(model, [sample])

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
            X, y, _ = batch
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

            # Calculating several batch statistics

            all_predictions.extend(outputs.detach().cpu().argmax(1).tolist())
            all_ground_truth_labels.extend(y.cpu().tolist())
            epoch_running_loss += loss.item()

        #scheduler.step()
        correct_list = [1 if a == b else 0 for a, b in zip(all_predictions, all_ground_truth_labels)]
        print(sum([1 for item in all_predictions if item != 0]))
        acc = sum(correct_list) / len(correct_list)
        prog_string = "[|Train| Loss: %.3f, Acc: %.3f, f_1: %.3f, recall: %.3f, precision, %.3f]" \
                      % (epoch_running_loss, acc,
                         f1_score(all_ground_truth_labels, all_predictions, average="weighted"),
                         recall_score(all_ground_truth_labels, all_predictions, average="weighted"),
                         precision_score(all_ground_truth_labels, all_predictions, average="weighted"))

        with open("%s/results.txt" % save_path, "a") as f:
            f.write(prog_string+"\n")
        if tensorboard_dir:
            writer.add_scalar('loss', epoch_running_loss, epoch)
            writer.add_scalar('accuracy', acc, epoch)

    print('Finished Training')
    torch.save(model.state_dict(), "%s/%s_epoch_%d.pt" % (save_path, save_name, epoch))
    if tensorboard_dir:
        writer.close()
    return model

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





