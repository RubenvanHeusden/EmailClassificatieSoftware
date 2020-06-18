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
when inputting the names of the column that you put them in the same folder as in the csv file (more info in this later).


Training the model
==================

