import pickle
from KNN import ActiveKNN
from SVM import ActiveSVM
from RandomForest import ActiveRandomForest
import lime
from lime import lime_tabular
import numpy as np

"""
Todo:
 - Need to think about Continuous, Categorical Features.
"""

all_method = ["KNN", "SVM"]
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

KNN_model, SVM_model = []
#Unpickling all the files
for method in all_method:
	for option in all_option:
		for amine in all_amines:
            directory_name = method + "_few_shot/" + option + method + "_20_shot_2_way_" + option + "_" + amine + "/"
            file_name = "KNN_" + amine + "_" + option + ".pkl"
            infile = open(directory_name + file_name, "rb")
            if method == 'KNN':
                KNN_model.append(pickle.load(infile))
            else:
                SVM_model.append(pickle.load(infile))

#infile = open("KNN_20_shot_2_way_option_2_BAMDIFIROXTEEM-UHFFFAOYSA-N/KNN_BAMDIFIROXTEEM-UHFFFAOYSA-N_option_2.pkl", "rb")
#activeKNN = pickle.load(infile)

#Define feature_names
feature_name = ["_rxn_M_acid", "_rxn_M_inorganic", "_rxn_M_organic", "_solv_GBL", "_solv_DMSO", "_solv_DMF","_stoich_mmol_org",	"_stoich_mmol_inorg", "_stoich_mmol_acid", "_stoich_mmol_solv",	"_stoich_org/solv",	"_stoich_inorg/solv","_stoich_acid/solv", "_stoich_org+inorg/solv","_stoich_org+inorg+acid/solv","_stoich_org/liq","_stoich_inorg/liq","_stoich_org+inorg/liq", "_stoich_org/inorg", "_stoich_acid/inorg", "_rxn_Temperature_C",	"_rxn_Reactiontime_s", "_feat_AvgPol", "_feat_Refractivity", "_feat_MaximalProjectionArea",	"_feat_MaximalProjectionRadius", "_feat_maximalprojectionsize", "_feat_MinimalProjectionArea",	"_feat_MinimalProjectionRadius", "_feat_minimalprojectionsize",	"_feat_MolPol",	"_feat_VanderWaalsSurfaceArea",	"_feat_ASA", "_feat_ASA_H", "_feat_ASA_P",	"_feat_ASA-",	"_feat_ASA+", "_feat_ProtPolarSurfaceArea", "_feat_Hacceptorcount", "_feat_Hdonorcount","_feat_RotatableBondCount",	"_raw_standard_molweight", "_feat_AtomCount_N", "_feat_BondCount",	"_feat_ChainAtomCount",	"_feat_RingAtomCount", "_feat_primaryAmine",	"_feat_secondaryAmine",	"_rxn_plateEdgeQ", "_feat_maxproj_per_N", "_raw_RelativeHumidity"]
class_name = ["failure", "success"]


for activeKNN in KNN_model:
    #Getting the KNN model, amine name, training data (x_true), training label (y_true).
    model = activeKNN.model
    amine = activeKNN.amine
    x_true = activeKNN.x_t
    y_true = activeKNN.y_t

    #Getting the prediction of the training data.
    y_pred = model.predict(x_true)
    y_pred_prob = model.predict_proba(x_true)
    
    #Getting the explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(x_true, feature_names = feature_name, class_names = class_name, discretize_continuous = True)

    #Explaining all instances

    for i in range(len(x_true)):
        exp = explainer.explain_instance(x_true[i], model.predict_proba, num_features = 10)
        lst_explanation = exp.as_list()

        print("Amine:", amine)
        print("Data:", str(i))
        print("Prediction:", class_name[y_pred[i]], "with probability", str(y_pred_prob[i][y_pred[i]]))
        print("True Class:", class_name[y_true[i]])
        print("Explanation:", lst_explanation)
        #fig_explanation = exp.as_pyplot_figure()
        #fig_explanation.savefig('Lime_' + amine + '_44.png')








