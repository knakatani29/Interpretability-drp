# as of July 31, 2020

# running l-shap_platipus.py creates figures from pt-files

# the pt-files are from Shekar running platipus 
# using the code at https://github.com/darkreactions/platipus/tree/shekar
# choosing the ones with the most training iterations (in this case 5,000)

# many of the requirements to run l-shap_platipus.py are included 
# in the platipus conda venv from Shekar, Vincent, and Sharon 
# if this venv is present, you can activate it using "conda activate platipus" 
# shap is not included in that, so to install shap, run either:
# "pip install shap" or "conda install -c conda-forge shap"

# the figures were created by running l-shap_platipus.py on the pt-files
# inside l-shap_platipus.py, can choose between models with and without the active learning 
# by changing which lines are commented 
