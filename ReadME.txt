Capstone Project - Readme
==================
Python Version: 3.6
Author: Bilal Shahid Cheema
Date: 8-22-2018


Description
----------------
This Python module gets stock prices from Yahoo finance, stores the data in the database and in a csv file.


Platforms
----------------
This application is platform agnostic.


Installation
--------------
1. Install Python 3.6.0 - https://www.python.org/downloads/ 
2. Unzip files to local drive in desired folder (example: C:\mini_project_i). 
3. Open cmd prompt / shell.
4. Navigate to created folder.
5. Install requirements:
   1. Type “pip install -r requirements.txt” in cmd prompt/shell.
   2. Install all requirements.
1. Run “main.py” by typing Python main.py in command prompt/shell.


Main Requirements
---------------------------
Python version 3.6 - See https://www.continuum.io/downloads for installation.
pip - Is included with Python 3.6. See https://pip.pypa.io/en/stable/installing/ for more. 


This module using the following Python Modules

Pandas - see https://pandas.pydata.org/pandas-docs/stable/ for more information.
NumPy - see https://docs.scipy.org/doc/ for more information.
datetime
time
math
fix_yahoo_finance
random
scipy
sklearn
matplotlib
quandl


import pandas as pd  # import pandas module for DataFrame, File reading and other functions
import datetime # import datatime for handling Dates
import numpy as np

Check requirements.txt for list of all other required sublibraries. Follow installation steps 
above to install all required libraries and modules.


Compatibility
-------------
The code has been tested in Puthon 3.6 but has been analyzed to be backwardly compatible to Python 2.7.


Module Installation
--------------------

Use the following command on command prompt to install a module if needed

pip install <modulename>