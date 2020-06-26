import pickle
from KNN import ActiveKNN
from SVM import ActiveSVM
from RandomForest import ActiveRandomForest
import shap
from lime import lime_tabular
import numpy as np
import matplotlib.pyplot as plt
import sys

"""
Todo:
 - Need to think about Continuous, Categorical Features.
"""

all_method = ["KNN", "RandomForest"]
all_option = ["option_1", "option_2"]
all_amines = ['ZEVRFFCPALTVDN-UHFFFAOYSA-N',
                    'KFQARYBEAKAXIC-UHFFFAOYSA-N',
                    'NLJDBTZLVTWXRG-UHFFFAOYSA-N',
                    'LCTUISCIGMWMAT-UHFFFAOYSA-N',
                    'JERSPYRKVMAEJY-UHFFFAOYSA-N',
                    'VAWHFUNJDMQUSB-UHFFFAOYSA-N',
                    'WGYRINYTHSORGH-UHFFFAOYSA-N',
                    'VNAAUNTYIONOHR-UHFFFAOYSA-N',
                    'FJFIJIDZQADKEE-UHFFFAOYSA-N',
                    'XFYICZOIWSBQSK-UHFFFAOYSA-N',
                    'UMDDLGMCNFAZDX-UHFFFAOYSA-O',
                    'HBPSMMXRESDUSG-UHFFFAOYSA-N',
                    'NXRUEVJQMBGVAT-UHFFFAOYSA-N',
                    'LLWRXQXPJMPHLR-UHFFFAOYSA-N',
                    'BAMDIFIROXTEEM-UHFFFAOYSA-N',
                    'XZUCBFLUEBDNSJ-UHFFFAOYSA-N']

KNN_model = []
RF_model = []
#Unpickling all the files
for method in all_method:
	for option in all_option:
		for amine in all_amines:
			directory_name = method + "_few_shot/" + option + "/" + method + "_20_shot_2_way_" + option + "_" + amine + "/"
			file_name = method + "_" + amine + "_" + option + ".pkl"
			infile = open(directory_name + file_name, "rb")
			if method == 'KNN':
				KNN_model.append(pickle.load(infile))
			else:
				RF_model.append(pickle.load(infile))

#infile = open("KNN_20_shot_2_way_option_2_BAMDIFIROXTEEM-UHFFFAOYSA-N/KNN_BAMDIFIROXTEEM-UHFFFAOYSA-N_option_2.pkl", "rb")
#activeKNN = pickle.load(infile)

#Define feature_names
feature_name = ["_rxn_M_acid", "_rxn_M_inorganic", "_rxn_M_organic", "_solv_GBL", "_solv_DMSO", "_solv_DMF","_stoich_mmol_org",	"_stoich_mmol_inorg", "_stoich_mmol_acid", "_stoich_mmol_solv",	"_stoich_org/solv",	"_stoich_inorg/solv","_stoich_acid/solv", "_stoich_org+inorg/solv","_stoich_org+inorg+acid/solv","_stoich_org/liq","_stoich_inorg/liq","_stoich_org+inorg/liq", "_stoich_org/inorg", "_stoich_acid/inorg", "_rxn_Temperature_C",	"_rxn_Reactiontime_s", "_feat_AvgPol", "_feat_Refractivity", "_feat_MaximalProjectionArea",	"_feat_MaximalProjectionRadius", "_feat_maximalprojectionsize", "_feat_MinimalProjectionArea",	"_feat_MinimalProjectionRadius", "_feat_minimalprojectionsize",	"_feat_MolPol",	"_feat_VanderWaalsSurfaceArea",	"_feat_ASA", "_feat_ASA_H", "_feat_ASA_P",	"_feat_ASA-",	"_feat_ASA+", "_feat_ProtPolarSurfaceArea", "_feat_Hacceptorcount", "_feat_Hdonorcount","_feat_RotatableBondCount",	"_raw_standard_molweight", "_feat_AtomCount_N", "_feat_BondCount",	"_feat_ChainAtomCount",	"_feat_RingAtomCount", "_feat_primaryAmine",	"_feat_secondaryAmine",	"_rxn_plateEdgeQ", "_feat_maxproj_per_N", "_raw_RelativeHumidity"]
class_name = ["failure", "success"]

open('out.txt', 'w')

model_list = [KNN_model, RF_model]
for model_index in range(2):
	for activemodel in model_list[model_index]:
		#Getting the model, amine name, training data (x_true), training label (y_true).
		model = activemodel.model
		amine = activemodel.amine
		x_true = activemodel.x_t
		y_true = activemodel.y_t

		#Getting the prediction of the training data.
		y_pred = model.predict(x_true)
		y_pred_prob = model.predict_proba(x_true)

        explainer = shap.KernelExplainer(model.predict_proba, x_true)
        shap_values = explainer.shap_values(x_true)
        shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)

        #Explaining a single instance
        #shap_values = explainer.shap_values(x_test.iloc[0,:])
        #shap.force_plot(explainer.expected_value[0], shap_values[0], X_test.iloc[0,:])  

		#for i in range(len(x_true)):
		# i = np.random.randint(0, len(x_true))
		# exp = explainer.explain_instance(x_true[i], model.predict_proba, num_features = 10)
		# lst_explanation = exp.as_list()
		
		# with open('out.txt', 'a') as f:

		# 	if model_index == 0:
		# 		print("Method: KNN", file = f)
		# 	else:
		# 		print("Method: Random Forest", file = f)
		# 	print("Amine:", amine, file = f)
		# 	print("Data:", str(i), file = f)
		# 	print("Prediction:", class_name[y_pred[i]], "with probability", str(y_pred_prob[i][y_pred[i]]), file = f)
		# 	print("True Class:", class_name[y_true[i]], file = f)
		# 	print("Explanation:", lst_explanation, file = f)
		# 	print("\n", file = f)
		
		# #fig_explanation = exp.as_pyplot_figure()
		
		# #Plotting the figure
		# fig = plt.figure()
		# vals = [x[1] for x in lst_explanation]
		# names = [x[0] for x in lst_explanation]
		# vals.reverse()
		# names.reverse()
		# colors = ['green' if x > 0 else 'red' for x in vals]
		# pos = np.arange(len(lst_explanation)) + .5
		# plt.barh(pos, vals, align='edge', color=colors)
		# plt.yticks(pos, names)
		# plt.title('Local Explanation')

		# if model_index == 0:
		# 	fig_name = "fig/Lime_KNN_" + amine + "_" + str(i) + ".png"
		# else:
		# 	fig_name = "fig/Lime_RF_" + amine + "_" + str(i) + ".png"
		# fig.savefig(fig_name, bbox_inches = 'tight')

		# plt.close()