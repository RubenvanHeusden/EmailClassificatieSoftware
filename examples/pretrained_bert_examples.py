"""
This file contains some examples about the  basic usage of the BERT model used in this research
"""
from configurations import ROOT_DIR
from models.pretrainedbert import PretrainedBERT


def main():
    # This examples script will show how to use the PretrainedBERT class available in this module
    # As the model training of the HuggingFace model is quite involved, this model attempts to bridge
    # this gap by providing a more high-level interface to the model, allowing training with commands very
    # similar to that used in the other models. Please note that training a BERT model requires a decent
    # GPU (minimum RAM of at least 6GB approx). A model pretrained on the toy dataset is provided for the
    # example in order to experiment with the BERT model and its functions.

    # First we will start by loading in the model, be aware that the model needs the label names of the dataset
    # for later use, so the path to the trainin file should also be specified

    model = PretrainedBERT(path_to_data=ROOT_DIR+"/test_data/train.csv")

    # Now that we have instantiated the PretrainedBERT model, we should start by loading the model weights
    # and tokenizer settings from the the folder created when the model is initially trained. As this model
    # has been pretrained on this dataset already, it already exists, it is the "bert_model_for_toy_dataset"
    # folder

    model.load_model(path_to_saved_model=ROOT_DIR+"/bert_model_for_toy_dataset")

    # Now that the model is loaded, we can do some classification with it on new datapoints, classify the
    # contents of a file, or score the model. Lets classify a small piece of text first.

    piece_of_text = "Ik vind dit eigenlijk best wel een hele goede film"

    prediction = model.classify_from_strings(piece_of_text)
    print(prediction)

    # Note that the prediction is actually wrong, this is not surprising though, consider we only trained
    # it on 3 sentences.

    # Now we can also classify an entire file if we want to, so let's try that one.

    file_predictions = model.classify_from_file(file_name=ROOT_DIR+"/test_data/test.csv")
    print(file_predictions)

    # Now if we would like to classify a list of strings we can use the 'classify_from_batches' method
    # You could also call 'classify_from_strings' on each item of the list of strings, however the
    # 'classify_from_batches' method batches the sentences in the lists and feeds them in batches to
    # the model, which is a lot faster than the 'classify_from_strings' methods, especially for
    # larger lists

    # Now if we want to see the actual performance of the model, we can also score its performance with
    # the 'score' function. This is currently only implemented for scoring of test files.

    model.score(file_name=ROOT_DIR+"/test_data/test.csv")

    # And that's it! Hopefully you will now have a basic understanind of how to use the BERT model
    # in practice!


if __name__ == "__main__":
    main()
