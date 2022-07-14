# # Example of overriding a variable defined in the SConstruct file:
# DATA_PATH = "/some/location/on/computer/"
# # Example of adding to a variable defined in the SConstruct file:
# EXPERIMENTS["my_new_experiment"] = {
# # experiment definition goes here
# }

#### Historical Text Experiments ####

EXPERIMENTS["documentary_hypothesis"] = {
    "variables" : {
                    "NUM_FOLDS" : 3,
                    "LANGUAGE" : "hebrew",
                    "CLASSIFIER_VALUES" : [("naive_bayes", "mlp")],
                    "FEATURE_SETS_VALUES" : ["BERT"],
                    "TARGET_CLASS_VALUES" : [("source")]
                }
    "data" : "${DATA_PATH}/reddit_style_transfer.json.gz"
}

EXPERIMENTS["tirant_lo_blanc"] : {
    "variables" : {
                    "NUM_FOLDS" : 3,
                    "LANGUAGE" : "catalan",
                    "TOKENS_PER_SUBDOCUMENT_VALUES" : [100, 400, 1600],
                    "CLASSIFIER_VALUES" : [],
                    "FEATURE_SETS_VALUES" : ["BERT"],
                    "TARGET_CLASS_VALUES" : []
                }
    "data" : "/exp/abeckwith/data/breakpoint/tirant_lo_blanc.txt"
}    
