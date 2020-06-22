import os
import pickle

import numpy as np
from KNN import ActiveKNN
from modAL.models import ActiveLearner
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

infile = open("KNN_few_shot/option_2/KNN_20_shot_2_way_option_2_BAMDIFIROXTEEM-UHFFFAOYSA-N/KNN_BAMDIFIROXTEEM-UHFFFAOYSA-N_option_2.pkl", "rb")
model = pickle.load(infile)



