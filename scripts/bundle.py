#!/usr/bin/env python3

import os
import shutil

# REQUIREMENTS: The following Modules must be installed somewhere in your system
#
import getopt
import joblib
import nltk
import numpy
import Stemmer
import sklearn

# COPY MODULES into $PWD.
# path = os.path.dirname(joblib.__file__)
def bundle():
    shutil.copytree(os.path.dirname(nltk.__file__), 'nltk')
    shutil.copytree(os.path.dirname(numpy.__file__), 'numpy')
    shutil.copytree(os.path.dirname(getopt.__file__), 'getopt')
    shutil.copytree(os.path.dirname(sklearn.__file__), 'sklearn')
    shutil.copytree(os.path.dirname(joblib.__file__), 'joblib')
    shutil.copytree(os.path.dirname(Stemmer.__file__), 'Stemmer')

if __name__ == '__main__':
    bundle()
