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
