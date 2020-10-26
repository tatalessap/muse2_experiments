from classification import *
from generate_features import *

"""
Generate features:
- one file
- one file per experiment
"""

generate_by_all_files('res_eti_256row/', 'feature_ex', False)

generate_by_one_file('res_eti_256row/', 'feature_ex', 'uni_features_smooth', True)

