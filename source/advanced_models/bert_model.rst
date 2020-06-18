The BERT model
**************

The Bidirectional Transformer (BERT) is a state-of-the-art model developed by Devlin et al. :cite:`devlin2018bert` at Google.
By using a Transformer model as a building block it is able to achieve very good performances on a variety of language tasks.
The model used in this research is developed by HuggingFace Transformers, who have also made several pretrained versions available,
including a Dutch Model. Normally, these pretrained models still have to be fine-tuned on a specific task, which is implemented in this 
package. Please note that the shear size of the BERT model means that a decent quality GPU is needed in order to train this model manually.

Fine-Tunining the model
=======================
We will first start by instantiating an instance of the BERT model that we will use for training.

.. code-block::

   bert_model = PretrainedBert(path_to_data="test_data/train.csv")

Now this model can either be used to fine-tune a model for classification or use an existing model.
As the training procedure for a BERT model can be quite involved, the :meth:`.PretrainedBert.train_from_file` method
uses default parameters for most options, and set to values that have shown to work well with the dataset used in the research.

Now if you have enough GPU power you can train the model by calling:

.. code-block::

		bert_model.train_from_file('test_data/', use_eval=True, num_epochs=1)
		
This will expect the 'test_data' folder to contain both a train.csv and test.csv file with 3 columns, namely
['idx', 'text', 'label']. if we set `use_eval=False` the model will not use the test.csv file and no performance statistics
will be calculated after training. Be aware that depending on the number of epochs you set, the training can take quite some time.
	
Now the loading of the model is slightly different than that from the TF-IDF, BiLSTM and CNN models.
Instead of the model automatically being loaded after calling the training command, the model still has to be loaded separately here.
This has to be done because of the specific way this is implemented in the HuggingFace code and the fact that the model needs extra
information stored in files created during training.

We will load the model after training:

.. code-block::

	bert_model.load_model(self, path_to_saved_model='bert_training/')


Now that we have loaded the model we can use it in much the same way as the other models! Let's try this out 
by classifying a sentence.

.. code-block::

	bert_model.classify_from_strings("Ik ben wel benieuwd naar dit nieuwe BERT model")
	

And that's it! Hopefully you are now a bit more familiar with the interface for working with the BERT model,
and don't forget to check out the example script for the model if you want to see some more examples.

