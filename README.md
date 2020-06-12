# EmailClassificatieSoftware

Deze repository bevat de code voor het Email Classificatie Systeem ontwikkeld door Ruben van Heusden voor de Gemeente Amsterdam.

## Inhoud

Het project bestaat uit verschillende modellen die kunnen worden gebruikt voor het classificeren van emails,
ook bevat de module enkele scripts voor eventuele anonimisatie van data en de mogelijkheid om de prestaties
van de modellen te evalueren.

### Working with word embeddings

The CNN and Bidirectional LSTM models both work with pretrained word embeddings as input. In the case of 
the Dutch Language, the word embeddings that are used in this research are the ... word embeddings
that can be found on ()

The training scripts for these models will automatically download these word embeddings when they are not 
found in the 'word_vectors' folder in the module, and will use those word embeddings from then onwards.



### Training Scripts

Voor de complexere modellen in deze module zijn aparte training scripts opgenomen die kunnen gebruikt
om de modellen te trainen op de gewenste data. Het getrainde model kan worden opgeslagen en daarna 
worden gebruikt voor classificatie.

### Training a BERT model

To finetune a pretrained BERT model on the Dutch classification task, the scripts in the 'bert_training_files'
folder are provided. Although it is recommended to use the model already trained during the research, there are 
cases in which the only option is to retrain the model, for example when changing the categories that need to be 
classified.

The easiest way to train the model is to put the data to be trained in a 'data' directory in the module
and let the Bert Model methods determine the labels and amount of classes based on the contents of the 'label' column
in the csv file.

To train the bert model, run the following command in a terminal when inside the EmailClassificatieSoftware package


`python bert_training_files/train_bert.py --task_name email_classification --do_train --do_eval --data_dir ../data --model_type bert --model_name_or_path bert-base-dutch-cased --overwrite_output_dir --learning_rate 2e-5 --per_gpu_train_batch_size 1 --num_train_epochs 10.0 --output_dir custom_intent_results --tensorboard_dir custom_intent_results/exp1  --logging_steps 500 --save_steps 12500 --max_seq_length 128 --gradient_accumulation_steps 100`

Although most settings can be left as is, it is recommended to lower the batch_size argument of the script when
running into memory problems.

The resulting output_folder containing the model files after training can be read in using the PretrainedBert
class from the models folder.