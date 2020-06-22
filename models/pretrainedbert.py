"""
This file implements a BERT model from the huggingface-transformers Library which is fine-tuned on the
specific dataset. As the training procedure is relatively involved, this model contains a wrapper for the training
of the model that uses a set of default parameter for training. For most use cases these default parameters are
good enough but if more control is needed they can be altered.
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
from typing import List
from configurations import ROOT_DIR
from sklearn.metrics import accuracy_score
from bert_training_files.train_bert import main
from transformers import BertForSequenceClassification, BertTokenizer


class PretrainedBERT:
    """
    This class implements an interface to a pretrained BERT model on Dutch data, handling the tokenization of the text
    and te conversion of the output of the model to the appropriate classes. This class can be used interact
    with a model trained using the Bert model in this module.

    Attributes
    ----------
    device: torch.device
        if use_gpu is set to true and a gpu is available on the system the model and inputs will be run on
        the gpu. This is significantly faster than running the model on the gpu.

    bert_model
        An instance of the BertForSequenceClassification model that is trained on the Dutch Language
        and used to classify the examples

    bert_tokenizer
        tokenizer for the BertModel this is trained on the Dutch language and used to convert sequences of strings
        to sequences of indices into the sentencepiece embeddings model of Bert.

    _class_labels
        labels that are used in this specific research, the labels should match the labels present in the
        dataset
    """
    def __init__(self, use_gpu: bool = False, path_to_data: str = ROOT_DIR+"/data/train.csv",
                 label_col_name: str = 'label'):
        """

        :param use_gpu: Boolean specifying whether or not to use the GPU. This only influences \
        the usage of the GPU during evaluation as the BERT model must be trained on a gpu to \
        be able to be trained in reasonable time.
        :param path_to_data: path specifying the where the train.csv data file is located
        :param label_col_name: parameter that indicates the name of the column where the labels of the \
        text are stored
        """
        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")

        self._class_labels = pd.read_csv(path_to_data, sep=",", quotechar='|')[label_col_name].unique()

        self.bert_model = None

        self.bert_tokenizer = None

    @staticmethod
    def train_from_file(path_to_datadir: str, max_seq_len: int = 128, output_dir: str = ROOT_DIR+'/bert_training/',
                        tensorboard_dir: str = ROOT_DIR+'/bert_runs/', do_eval: bool = False,
                        num_epochs: int = 10) -> None:
        """
        This method can be used to train the bert model. It acts as a wrapper around the code in the
        'train_bert.py' file and sets most parameters to a decent default value. If more control is needed the
        specific functions of the parameters in the model can be found in the documentation of the Huggingface package.

        :param path_to_datadir: string specifying the path to the data directory. Here the train file is read in \
        and used for training. By default this method does not perform evaluation so there is no need for a 'test.csv'\
         file however if do_eval is set to True is possible. It is important to note that because of the specifics \
         of the huggingface package the file(s) should be called 'train.csv' (and 'test.csv' if do_eval = True)
        :param max_seq_len: Integer specifying the maximum length to which all the sentences are truncated. If memory \
        issues occur, it is possible to set this parameter to a lower value, at the cost of reduced performance of \
        the model
        :param output_dir: string specific the path where the output of the model should be stored (containing) \
        the specific training command used and the saved version of the trained model.
        :param tensorboard_dir: string specific the directory where the tensorboard log file should be stored.
        :param do_eval: Boolean specifying whether or not to evaluate the performance of the model on a test set \
        after training.
        :param num_epochs: int specifying for how many epochs the model should be trained.
        :return: None
        """
        args = {'data_dir': path_to_datadir, 'model_type': 'bert', 'model_name_or_path': 'bert-base-dutch-cased',
                'task_name': 'email_classification', 'output_dir': output_dir, 'config_name': "",
                'tokenizer_name': "", 'cache_dir': "", 'max_seq_length': max_seq_len, 'do_train': True,
                'do_eval': do_eval, 'evaluate_during_training': False, 'do_lower_case': False,
                'per_gpu_train_batch_size': 8, 'per_gpu_eval_batch_size': 8, 'gradient_accumulation_steps': 1,
                'learning_rate': 2e-5, 'weight_decay': 0.0, 'adam_epsilon': 1e-8, 'max_grad_norm': 1.0,
                'num_train_epochs': num_epochs, 'max_steps': -1, 'warmup_steps': 0,
                'tensorboard_dir': tensorboard_dir, 'logging_steps': 500, 'save_steps': 500,
                'eval_all_checkpoints': False, 'no_cuda': False, 'overwrite_output_dir': True,
                'overwrite_cache': True, 'seed': 42, 'fp16': False, 'fp16_opt_level': "0.1", 'local_rank': -1,
                'server_ip': "", 'server_port': "", 'n_gpu': 1}
        if (
                os.path.exists(args['output_dir'])
                and os.listdir(args['output_dir'])
                and args['do_train']
                and not args['overwrite_output_dir']
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    args['output_dir']
                )
            )
        # Run the main train command from the 'train_bert.py' file
        main(args)
        return None

    def classify_from_strings(self, text_string: str) -> List:
        """

        Method that can be used to classify a string or a list of strings with the BERT Model.
        This does require that the model used for classification is first loaded via the
        'load_model' method

        :param text_string: string that contains the text to be classified
        :return: list containing the string representation of the most likely class according \
        to the Bert model.
        """

        assert self.bert_model

        input_ids = torch.tensor([self.bert_tokenizer.encode(text_string, add_special_tokens=True)]).to(self.device)
        with torch.no_grad():
            model_output = self.bert_model(input_ids)
            return self._class_labels[model_output[0].argmax()]

    def classify_from_file(self, file_name: str, delimiter: str = ",", quotechar: str = '"',
                           text_col_name: str = 'text', batch_size: int = 8) -> List:
        """
        Convenience function to classify the contents of a csv file. This function will call
        'classify_batches' internally to do the classification

        :param file_name: string indicating the csv file containing the text to be classified
        :param delimiter: delimiter used in the csv file, to read the csv file
        :param quotechar: quotechar used in the csv file, to read  the csv file
        :param text_col_name: string signifying the name of the column in the csv file containing the \
        text to be classified
        :param batch_size: amount of examples to feed to the network simultaneously, this \
        can be lowered if memory issues are encountered
        :return: list of predictions for the text found in the csv file
        """

        assert self.bert_model

        text_from_file = pd.read_csv(file_name, sep=delimiter, quotechar=quotechar)[text_col_name]
        text_from_file = text_from_file.dropna().tolist()
        batches = [text_from_file[i:i + batch_size] for i in range(0, len(text_from_file), batch_size)]
        predictions = []
        for batch in tqdm(batches):
            predictions.extend(self.classify_batches(batch))
        return predictions

    def classify_batches(self, list_of_examples: List) -> List[str]:
        """
        This method implements a way of classifying a list of examples by feeding batches into the network.
        This is significantly more efficient for classifying a list of examples than calling the
        "classify_from_strings" method on each item in the list

        :param list_of_examples: list of strings containing the texts to be classified by the Bert model
        :return: list of predictions with the same length as the number of items in 'list_of_examples'
        """

        assert self.bert_model

        encoded_batch = self.bert_tokenizer.batch_encode_plus(list_of_examples, add_special_tokens=True,
                                                              return_tensors='pt', pad_to_max_length=True,
                                                              max_length=100)
        with torch.no_grad():
            model_output = self.bert_model(encoded_batch['input_ids'].to(self.device),
                                           attention_mask=encoded_batch['attention_mask'].to(self.device))
            predicted_classes = model_output[0].argmax(1)

            return [self._class_labels[i] for i in predicted_classes]

    def load_model(self, path_to_saved_model: str = 'bert-base-dutch-cased') -> None:
        """
        Method that can be used to load a trained model. The PretrainedBert class saves models that
        are trained using 'train_from_files' automatically so there is no separate method for model saving.

        :param path_to_saved_model: path to the directory containing the files created by the 'train_from_files' \
        method
        :return: None
        """

        self.bert_model = BertForSequenceClassification.from_pretrained(path_to_saved_model).to(self.device)
        self.bert_tokenizer = BertTokenizer.from_pretrained(path_to_saved_model)

        return None

    def score(self, file_name: str, delimiter: str = ",", quotechar: str = '"', text_col_name: str = 'text',
              label_col_name: str = 'label', batch_size: int = 8) -> None:
        """
        Function that can be used to test the performance of the model on an unseen test set.
        This function will call 'classify_batches' internally to do the classification

        :param file_name: string indicating the csv file containing the text to be classified
        :param delimiter: delimiter used in the csv file, to read the csv file
        :param quotechar: quotechar used in the csv file, to read  the csv file
        :param text_col_name: string signifying the name of the column in the csv file containing the \
        text to be classified
        :param label_col_name: string signifying the name of the column in the csv file containing the \
        labels of the text to be classified
        :param batch_size: amount of examples to feed to the network simultaneously, this \
        can be lowered if memory issues are encountered
        :return: None
        """

        assert self.bert_model

        file_contents = pd.read_csv(file_name, sep=delimiter, quotechar=quotechar).dropna()
        text_from_file = file_contents[text_col_name].tolist()
        labels_from_file = file_contents[label_col_name].tolist()
        batches = [text_from_file[i:i + batch_size] for i in range(0, len(text_from_file), batch_size)]
        predictions = []
        for batch in tqdm(batches):
            predictions.extend(self.classify_batches(batch))
        accuracy = accuracy_score(labels_from_file, predictions)
        print(accuracy)
