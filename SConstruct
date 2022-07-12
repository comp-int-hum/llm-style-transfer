import os
import steamroller
from glob import glob

vars = Variables("custom.py")
vars.AddVariables(
    (
        "OUTPUT_WIDTH",
        "Truncate the display length of commands after this number of characters",
        400
    ),
    (
        "DATA_PATH",
        "The location where data sets can be found (see how '${DATA_PATH}' is used below)",
        "data"
    ),
    (
        "TRAIN_DEV_TEST_PROPORTIONS",
        "Proportion of data splits for supervised training",
        (0.8, 0.1, 0.1)
    ),
    (
        "EXPERIMENTS",
        "Define experiments in this dictionary",
        {            
            "the_woman_of_colour" : {
                "variables" : {
                    "NUM_FOLDS" : 3,
                    "LANGUAGE" : "english",
                    "TOKENS_PER_SUBDOCUMENT_VALUES" : [100, 400, 1600],
                    "CLASSIFIER_VALUES" : ["naive_bayes"],
                    "FEATURE_SETS_VALUES" : ["stopwords_from_nltk"],
                    "TARGET_CLASS_VALUES" : [("author",)],
                },
                "data" : "${DATA_PATH}/woman_of_colour.tgz",
            },
            "reddit_style_transfer" : {
                "variables" : {
                    "NUM_FOLDS" : 3,
                    "LANGUAGE" : "english",
                    "TOKENS_PER_SUBDOCUMENT_VALUES" : [100, 400, 1600],
                    "CLUSTER_COUNT_VALUES" : [5, 10],
                    "CLASSIFIER_VALUES" : ["naive_bayes"],
                    "FEATURE_SETS_VALUES" : ["stopwords_from_nltk"],
                    "TARGET_CLASS_VALUES" : [("author", "original_author", "style")],                    
                },
                "data" : "${DATA_PATH}/reddit_style_transfer.json.gz",
            }            
        }
    )
)

env = Environment(
    ENV=os.environ,
    variables=vars,
    tools=[steamroller.generate],
    BUILDERS={
        "ExtractDocuments" : Builder( # turns various formats into a simple JSON object with a "text" and "author" fields
            action="python scripts/extract_documents.py --primary_sources ${SOURCE} --documents ${TARGET} --language ${LANGUAGE}"
        ), 
        #"ModifyDocument" : Builder( # takes an original document and changes *just the text* in some interesting way
        #    action="python scripts/modify_documents.py --original ${SOURCE} --modified ${TARGET} --modification_method ${MODIFICATION_METHOD}"
        #),
        "DivideDocuments" : Builder( # splits a JSON object's "text" field into a list of subdocuments
            action="python scripts/divide_document.py --documents ${SOURCE} --subdocuments ${TARGET} --tokens_per_subdocument ${TOKENS_PER_SUBDOCUMENT}"
        ),
        "ExtractRepresentations" : Builder( # extracts some number of (stylometric) features for each sub-document, using the specified method
            action="python scripts/extract_representations.py --subdocuments ${SOURCES} --representations ${TARGET}"
        ),
        "ClusterRepresentations" : Builder( # performs k-means clustering of (sub)-document representations
            action="python scripts/cluster_representations.py --representations ${SOURCE} --clustering ${TARGET} --cluster_count ${CLUSTER_COUNT}"
        ),
        "SplitData" : Builder( # performs k-means clustering of (sub)-document representations
            action="python scripts/split_data.py --representations ${SOURCE} --random_seed ${RANDOM_SEED} --train_dev_test_proportions ${TRAIN_DEV_TEST_PROPORTIONS} --train ${TARGETS[0]} --dev ${TARGETS[1]} --test ${TARGETS[2]}"
        ),
        "TrainClassifier" : Builder( # trains and serializes a (Naive Bayes?) classifier from features to author
            action="python scripts/train_classifier.py --train ${SOURCES[0]} --dev ${SOURCES[1]} --classifier ${CLASSIFIER} --feature_sets ${FEATURE_SETS} --target_class ${TARGET_CLASS} --model ${TARGET}"
        ),
        "ApplyClassifier" : Builder( # applies a trained classifier to given representations
            action="python scripts/apply_classifier.py --model ${SOURCES[0]} --test ${SOURCES[1]} --feature_sets ${FEATURE_SETS} --target_class ${TARGET_CLASS} --results ${TARGET}"
        ),
        "SaveConfiguration" : Builder( # saves the experimental configuration at the current moment (a hack)
            action="python scripts/save_configuration.py --configuration ${TARGET} --tokens_per_subdocument ${TOKENS_PER_SUBDOCUMENT} --feature_sets ${FEATURE_SETS} --fold ${FOLD} --target_class ${TARGET_CLASS}"
        ),
        # "EvaluateClassifications" : Builder( # performs some evaluation of classification performance (heatmap?)
        #    action="python scripts/evaluate_classifications.py --summary ${TARGET} ${SOURCES}"
        # ),
        "EvaluateClassifications" : Builder(
           action="python scripts/evaluate_classifications.py --summary ${TARGET} ${SOURCES}"
        )
    }
)


def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print(s[:int(float(env["OUTPUT_WIDTH"]) / 2) - 2] + "..." + s[-int(float(env["OUTPUT_WIDTH"]) / 2) + 1:])
    else:
        print(s)
env['PRINT_CMD_LINE_FUNC'] = print_cmd_line


for experiment_name, experiment in env["EXPERIMENTS"].items():

    full_documents = env.ExtractDocuments(
        "work/${EXPERIMENT_NAME}/full_documents.json.gz",
        experiment["data"],
        EXPERIMENT_NAME=experiment_name,
        LANGUAGE=experiment["variables"].get("LANGUAGE", "english"),
    )

    classification_outputs = []
    clustering_outputs = []
    for tokens_per_subdocument in experiment["variables"].get("TOKENS_PER_SUBDOCUMENT_VALUES", [None]):
        subdocuments = env.DivideDocuments(
            "work/${EXPERIMENT_NAME}/${TOKENS_PER_SUBDOCUMENT}/subdocuments.json.gz",
            full_documents,
            EXPERIMENT_NAME=experiment_name,
            TOKENS_PER_SUBDOCUMENT=tokens_per_subdocument
        )
        representations = env.ExtractRepresentations(
            "work/${EXPERIMENT_NAME}/${TOKENS_PER_SUBDOCUMENT}/representations.json.gz",
            subdocuments,
            EXPERIMENT_NAME=experiment_name,
            TOKENS_PER_SUBDOCUMENT=tokens_per_subdocument,
        )
        for fold in range(experiment["variables"]["NUM_FOLDS"]):
            train, dev, test = env.SplitData(
                ["work/${EXPERIMENT_NAME}/${TOKENS_PER_SUBDOCUMENT}/${FOLD}/%s.json.gz" % split for split in ["train", "dev", "test"]],
                representations,
                EXPERIMENT_NAME=experiment_name,
                TOKENS_PER_SUBDOCUMENT=tokens_per_subdocument,
                RANDOM_SEED=fold,
                FOLD=fold
            )
            for feature_sets in experiment["variables"]["FEATURE_SETS_VALUES"]:
                for classifier in experiment["variables"]["CLASSIFIER_VALUES"]:
                    for target_class in experiment["variables"]["TARGET_CLASS_VALUES"]:
                        model = env.TrainClassifier(
                            "work/${EXPERIMENT_NAME}/${TOKENS_PER_SUBDOCUMENT}/${FOLD}/${CLASSIFIER}/${FEATURE_SETS_COMPONENTS}/${TARGET_CLASS_COMPONENT}/model.pkl.gz",
                            [train, dev],
                            EXPERIMENT_NAME=experiment_name,
                            TOKENS_PER_SUBDOCUMENT=tokens_per_subdocument,
                            RANDOM_SEED=fold,
                            FOLD=fold,
                            CLASSIFIER=classifier,
                            FEATURE_SETS=feature_sets,
                            FEATURE_SETS_COMPONENT="_".join([x for x in sorted(feature_sets)]),
                            TARGET_CLASS=target_class,
                            TARGET_CLASS_COMPONENT="_".join([x for x in sorted(target_class)])
                        )
                        output = env.ApplyClassifier(
                            "work/${EXPERIMENT_NAME}/${TOKENS_PER_SUBDOCUMENT}/${FOLD}/${CLASSIFIER}/${FEATURE_SETS_COMPONENTS}/${TARGET_CLASS_COMPONENT}/output.json.gz",
                            [model, test],
                            EXPERIMENT_NAME=experiment_name,
                            TOKENS_PER_SUBDOCUMENT=tokens_per_subdocument,
                            RANDOM_SEED=fold,
                            FOLD=fold,
                            CLASSIFIER=classifier,
                            FEATURE_SETS=feature_sets,
                            FEATURE_SETS_COMPONENT="_".join([x for x in sorted(feature_sets)]),
                            TARGET_CLASS=target_class,
                            TARGET_CLASS_COMPONENT="_".join([x for x in sorted(target_class)])
                        )
                        config = env.SaveConfiguration(
                            "work/${EXPERIMENT_NAME}/${TOKENS_PER_SUBDOCUMENT}/${FOLD}/${CLASSIFIER}/${FEATURE_SETS_COMPONENTS}/${TARGET_CLASS_COMPONENT}/config.json.gz",
                            [],
                            EXPERIMENT_NAME=experiment_name,
                            TOKENS_PER_SUBDOCUMENT=tokens_per_subdocument,
                            RANDOM_SEED=fold,
                            FOLD=fold,
                            CLASSIFIER=classifier,
                            FEATURE_SETS=feature_sets,
                            FEATURE_SETS_COMPONENT="_".join([x for x in sorted(feature_sets)]),
                            TARGET_CLASS=target_class,
                            TARGET_CLASS_COMPONENT="_".join([x for x in sorted(target_class)])
                        )
                        classification_outputs.append((config, output, model, train, dev, test))
                
    #                 for cluster_count in experiment["variables"].get("CLUSTER_COUNT_VALUES", [10]):
    #                     clustering = env.ClusterRepresentations(
    #                         "work/${EXPERIMENT_NAME}/${FOLD}/${LOWERCASE}/${TOKENS_PER_SUBDOCUMENT}/${FEATURE_SELECTION_METHOD}/${NUM_FEATURES_TO_KEEP}/${CLUSTER_COUNT}/clustering.json",                            
    #                         representations,
    #                         EXPERIMENT_NAME=experiment_name,
    #                         FOLD=fold,
    #                         LOWERCASE=lowercase,
    #                         NUM_FEATURES_TO_KEEP=num_features_to_keep,
    #                         TOKENS_PER_SUBDOCUMENT=tokens_per_subdocument,
    #                         FEATURE_SELECTION_METHOD="stopwords",
    #                         CLUSTER_COUNT=cluster_count
    #                     )
    #                     configuration = env.SaveConfiguration(
    #                         "work/${EXPERIMENT_NAME}/${FOLD}/${LOWERCASE}/${TOKENS_PER_SUBDOCUMENT}/${FEATURE_SELECTION_METHOD}/${NUM_FEATURES_TO_KEEP}/${CLUSTER_COUNT}/configuration.json",
    #                         [],
    #                         EXPERIMENT_NAME=experiment_name,
    #                         FOLD=fold,
    #                         LOWERCASE=lowercase,
    #                         NUM_FEATURES_TO_KEEP=num_features_to_keep,
    #                         TOKENS_PER_SUBDOCUMENT=tokens_per_subdocument,
    #                         FEATURE_SELECTION_METHOD="stopwords",
    #                         CLUSTER_COUNT=cluster_count
    #                     )                    
    #                     results.append((clustering, configuration))
    # clustering_summary = env.EvaluateClusterings(
    #     "work/${EXPERIMENT_NAME}/clustering_summary.png",
    #     results,
    #     EXPERIMENT_NAME=experiment_name,
    # )
    classificiation_summary = env.EvaluateClassifications(
        "work/${EXPERIMENT_NAME}/classification_summary.png",
        classification_outputs,
        EXPERIMENT_NAME=experiment_name,
    )
    
