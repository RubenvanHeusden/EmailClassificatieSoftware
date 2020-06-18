The Multigate Mixture-of-Experts(MMoE) model
********************************************

As per of the research conducted, a Multigate Mixture-of-Experts model was implemented and tested
on a dataset supplied by the Gemeente. Although the performance of the model is not better than that 
of simpler models for this specific task, it is still included in the package because it is an interesting
model from a research perspective and it might perform better on a different set of tasks not test in this research.



Notes on the Data
=================

In short, a Multitask model attempts to learn several tasks at the same time. In the case of text classification,
this could mean predicting the category and emotion of a piece of text at the same time. This means that if you want 
to use this model your dataset should include different tasks and labels for these tasks.

When reading in the data from the file the model will look for the names of the task headers in order, so be careful
when inputting the names of the column that you put them in the same folder as in the csv file.


Training the model
==================

If you have set up the data properly as described above, training the model is actually quite easy.
As usual, we will start by setting up the classifier and setting some trianing parameters.

.. code-block::

	model = MultigateModel(num_outputs_list=[2, 5, 5], target_names_list=["label", "col_a", "col_b"], max_seq_len=10,
	n_experts=3)

As mentioned before it is important that the column headers you give to the 'target_names_list' appear in the same order 
as in the file, this also holds for the 'num_outputs_list', which is just the number of unique classes for each task.
Just like with the CNN and LSTM models, if you don't know these in advance you can just use the 'get_num_labels_from_file'
methods from the utils file to get the correct number of labels for each task. The 'n_experts' parameter indicates how 
many models are used in the 'shared bottom layer', the more you choose here the bigger the model gets, meaning it is slower
to train.

Now we are ready to train the model!

.. code-block::

	model.train_from_file(file_name="test_data/multicol_train.csv")


Now that we have train the model, we can just use the same methods available to us as with the simpler models to make some 
predictions or saved the model if we are happy with it.

.. code-block::

	model.classify_from_strings(["a", "dit is nog een test", "laatse zin"])
	model.save_model('awesome_multitask_model.pt')