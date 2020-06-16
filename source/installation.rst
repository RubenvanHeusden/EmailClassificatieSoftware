Installation
************


The EmailClassificatieSoftware package can be installed either via pip or anaconda.
The package has only been tested on a Windows 10 operating system but is expected
to work on Linux and Mac based systems as well. For both the pip and Anaconda installations 
it is highly recommended to create a (new) virtual environment for the project and install 
this package in that virtual environment.

Installation via Pip
====================
To install the the dependencies of the package on Windows, open the Command Prompt and navigate to the root directory
of the EmailClassificatieSoftware package.

Now run the following command in the terminal of the Command Prompt:

.. code-block:: python

	pip install -r requirements.txt

This will install the dependencies required for the package to run properly. Please
be aware that this also downloads the PyTorch package from their official website directly
instead of via pip. 


Installation via Anaconda
=========================

As mentioned above, it is highly recommended to install this pacakge into a virtual environment.
An easy way to do this is by using Anaconda. If you do not have Anaconda installed, you can download it 
from their website (www.anaconda.com). After downloading the package from GitHub and with Anaconda install,
open an Anaconda prompt and navigate to the root directory of this package. Now to create a new environment 
with the appropriate dependencies installed, type the following into the command line:

.. code-block:: python

	conda env create -f environment.yml

This will install the requirements listed in the 'environment.yml' file and create a new environment called 
'emailclassificatiesoftware'.


