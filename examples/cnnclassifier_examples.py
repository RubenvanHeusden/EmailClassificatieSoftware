from configurations import ROOT_DIR
from models.cnnclassifier import CNNClassifier


def main():
    # first we will initialize the CNN classifier that we will used for classification.
    # In the case of this example we know that the amount of labels present in the dataset is 2
    # so we can give that to the classifier via the 'num_outputs' argument
    # As the sentences we are classifying are quite short we can set the maximum length a bit higher
    # so that they all will be padded to fit the model
    classifier = CNNClassifier(num_outputs=2, max_seq_len=10)

    # The next step is to train the model from a file, which we can do with the
    # 'train_from_file' method. Here we can give in some train specific parameters for the model.
    # Because we are only using the small dataset as an example we will use a batch_size of 1 and just
    # Train for 5 epochs as an example.
    classifier.train_from_file(file_name=ROOT_DIR+"/test_data/train.csv", batch_size=1, num_epochs=5)

    # Now that we have trained the classifier we can now use it do classify examples from text

    print(classifier.classify_from_strings("Dit is een kleine test zin"))


if __name__ == "__main__":
    main()
