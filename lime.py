import pickle
from KNN import ActiveKNN

import lime

#Unpickling the model
infile = open("KNN_20_shot_2_way_option_2_BAMDIFIROXTEEM-UHFFFAOYSA-N/KNN_BAMDIFIROXTEEM-UHFFFAOYSA-N_option_2.pkl", "rb")
activeKNN = pickle.load(infile)

model = activeKNN.model
amine = activeKNN.amine

