
import pickle
from KNN import ActiveKNN
import lime
from lime import lime_tabular
import numpy as np


#Unpickling the model
infile = open("KNN_20_shot_2_way_option_2_BAMDIFIROXTEEM-UHFFFAOYSA-N/KNN_BAMDIFIROXTEEM-UHFFFAOYSA-N_option_2.pkl", "rb")
activeKNN = pickle.load(infile)

#Getting the KNN model, amine name, training data, and training label.
model = activeKNN.model
amine = activeKNN.amine
x_true = activeKNN.x_t
y_true = activeKNN.y_t
metrics = activeKNN.metrics

x_test = activeKNN.x_v
y_test = activeKNN.y_v

x_all = activeKNN.pool_data

#Getting the prediction of the training data.
y_pred = model.predict(x_true)
y_pred_prob = model.predict_proba(x_true)

"""
print(y_true)
print(y_pred)
print(y_pred_prob)
print(metrics)
print(x_true[0])
"""

#Need to check if feature_names are right
feature_names = ["_rxn_M_acid", "_rxn_M_inorganic", "_rxn_M_organic", "_solv_GBL", "_solv_DMSO", "_solv_DMF","_stoich_mmol_org",	"_stoich_mmol_inorg", "_stoich_mmol_acid", "_stoich_mmol_solv",	"_stoich_org/solv",	"_stoich_inorg/solv","_stoich_acid/solv", "_stoich_org+inorg/solv","_stoich_org+inorg+acid/solv","_stoich_org/liq","_stoich_inorg/liq","_stoich_org+inorg/liq", "_stoich_org/inorg", "_stoich_acid/inorg", "_rxn_Temperature_C",	"_rxn_Reactiontime_s", "_feat_AvgPol", "_feat_Refractivity", "_feat_MaximalProjectionArea",	"_feat_MaximalProjectionRadius", "_feat_maximalprojectionsize", "_feat_MinimalProjectionArea",	"_feat_MinimalProjectionRadius", "_feat_minimalprojectionsize",	"_feat_MolPol",	"_feat_VanderWaalsSurfaceArea",	"_feat_ASA", "_feat_ASA_H", "_feat_ASA_P",	"_feat_ASA-",	"_feat_ASA+", "_feat_ProtPolarSurfaceArea", "_feat_Hacceptorcount", "_feat_Hdonorcount","_feat_RotatableBondCount",	"_raw_standard_molweight", "_feat_AtomCount_N", "_feat_BondCount",	"_feat_ChainAtomCount",	"_feat_RingAtomCount", "_feat_primaryAmine",	"_feat_secondaryAmine",	"_rxn_plateEdgeQ", "_feat_maxproj_per_N", "_raw_RelativeHumidity"]


explainer = lime.lime_tabular.LimeTabularExplainer(x_true, feature_names = feature_names, class_names = ["success", "failure"], discretize_continuous = True)

"""
Explaining an Instance
"""



"""
print(len(x_all))
print(x_true)
print(len(x_true))
print(x_test)
print(len(x_test))
"""
#Using 44 because we know y_true[44] = 1
#i = np.random.randint(0, len(x.all))
exp = explainer.explain_instance(x_true[44], model.predict_proba, num_features = 10)

lst_explanation = exp.as_list()
fig_explanation = exp.as_pyplot_figure()
print(lst_explanation)
fig_explanation.savefig('Example_lime.png')





