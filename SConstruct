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
        "EXPERIMENTS",
        "Define experiments in this dictionary",
        {
            "the_woman_of_colour" : {
                "variables" : {
                    "NUM_FOLDS" : 3,
                    "WORDS_PER_SUBDOCUMENT_VALUES" : [100, 400, 1600],
                    "FEATURE_SELECTION_METHOD_VALUES" : ["stopwords"],
                    "NUM_FEATURES_TO_KEEP_VALUES" : [10, 40, 80, 160],
                    "CLUSTER_COUNT_VALUES" : [5, 10],
                    "LOWERCASE_VALUES" : [False],
                    "MODIFICATION_METHOD_VALUES" : [],                    
                },
                "data" : {
                    "UNMODIFIED_DOCUMENTS" : ["data/woman_of_colour/*tei"],
                    "MODIFIED_DOCUMENTS" : {},
                }
            },
            "styll 3x3" : {
                "variables": {
                    "NUM_FOLDS" : 3,
                    "FEATURE_SELECTION_METHOD_VALUES" : ["stopwords"],
                    "NUM_FEATURES_TO_KEEP_VALUES" : [10, 40, 80, 160],
                    "LOWERCASE_VALUES" : [False],
                },
                "data" : {
                    "UNMODIFIED_DOCUMENTS" : ["/exp/apatel/style_transfer/style_transfer_datasets/reddit_test_query/random/source_author_posts/"]
                }

            }

        }
    )
)

env = Environment(
    ENV=os.environ,
    variables=vars,
    tools=[steamroller.generate],
    BUILDERS={
        "ExtractDocument" : Builder( # turns various formats into a simple JSON object with a "text" and "author" fields
            action="python scripts/extract_document.py --primary_source ${SOURCE} --document ${TARGET}"
        ), 
        #"ModifyDocument" : Builder( # takes an original document and changes *just the text* in some interesting way
        #    action="python scripts/modify_documents.py --original ${SOURCE} --modified ${TARGET} --modification_method ${MODIFICATION_METHOD}"
        #),
        "DivideDocuments" : Builder( # splits a JSON object's "text" field into a list of subdocuments
            action="python scripts/divide_document.py --document ${SOURCE} --subdocuments ${TARGET} --words_per_subdocument ${WORDS_PER_SUBDOCUMENT}"
        ),
        "ExtractRepresentations" : Builder( # extracts some number of (stylometric) features for each sub-document, using the specified method
            action="python scripts/extract_representations_orig.py --subdocuments ${SOURCES} --representations ${TARGET} ${'--lowercase' if LOWERCASE else ''} --feature_selection_method ${FEATURE_SELECTION_METHOD} --num_features_to_keep ${NUM_FEATURES_TO_KEEP}"
        ),
        "ClusterRepresentations" : Builder( # performs k-means clustering of (sub)-document representations
            action="python scripts/cluster_representations.py --representations ${SOURCE} --clustering ${TARGET} --cluster_count ${CLUSTER_COUNT}"
        ),
        "TrainClassifier" : Builder( # trains and serializes a (Naive Bayes?) classifier from features to author
            action="python scripts/train_classifier.py --representations ${SOURCE} --model ${TARGET}"
        ),
        # "ApplyClassifier" : Builder( # applies a trained classifier to given representations
        #     action="python scripts/apply_classifier.py --model ${SOURCES[0]} --representations ${SOURCES[1]} --results ${TARGET}"
        # ),
        "SaveConfiguration" : Builder( # saves the experimental configuration at the current moment (a hack)
            action="python scripts/save_configuration.py --configuration ${TARGET} --cluster_count ${CLUSTER_COUNT} --lowercase ${LOWERCASE} --words_per_subdocument ${WORDS_PER_SUBDOCUMENT} --feature_selection_method ${FEATURE_SELECTION_METHOD} --num_features_to_keep ${NUM_FEATURES_TO_KEEP} --fold ${FOLD}"
        ),
        # "EvaluateClassifications" : Builder( # performs some evaluation of classification performance (heatmap?)
        #    action="python scripts/evaluate_classifications.py --summary ${TARGET} ${SOURCES}"
        # ),
        "EvaluateClusterings" : Builder( # plots adjusted mutual information of clusterings w.r.t. known authorship, against experimental settings
            action="python scripts/evaluate_clusterings.py --summary ${TARGET} ${SOURCES}"
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

    # Marshall all of the relevant data for the experiment
    unmodified_documents = [env.ExtractDocument(
        "work/${EXPERIMENT_NAME}/data/unmodified/${SOURCE.name}.json",
        p
    ) for p in sum([Glob(x) for x in experiment["data"]["UNMODIFIED_DOCUMENTS"]], [])]
    if len(unmodified_documents) == 0:
        env.Exit(
            "No unmodified documents: make sure there are files underneath 'data/' that match values in the experiment's UNMODIFIED_DOCUMENTS variable"
        )
    modified_documents = {
        k : sum([Glob(x) for x in v], []) for k, v in experiment["data"]["MODIFIED_DOCUMENTS"].items()
    }
    for modification_method in experiment["variables"]["MODIFICATION_METHOD_VALUES"]:        
        modified_documents[modification_method] = []
        for unmod_doc in unmodified_documents:
            modified_documents[modification_method].append(
                env.ModifyDocuments(
                    "work/${EXPERIMENT_NAME}/modified/${MODIFICATION_METHOD}.json",
                    unmod_doc,
                    EXPERIMENT_NAME=experiment_name,
                    MODIFICATION_METHOD=modification_method
                )
            )

    # Run and evaluate basic clustering
    results = []
    for fold in range(1, experiment["variables"].get("NUM_FOLDS", 1) + 1):
        for lowercase in experiment["variables"].get("LOWERCASE_VALUES", [True]):
            for words_per_subdocument in experiment["variables"].get("WORDS_PER_SUBDOCUMENT_VALUES", [2000]):
                all_subdocuments = []
                for document in unmodified_documents:
                    subdocuments = env.DivideDocuments(
                        "work/${EXPERIMENT_NAME}/${FOLD}/${LOWERCASE}/${WORDS_PER_SUBDOCUMENT}/${SOURCE.name}.json",
                        document,
                        EXPERIMENT_NAME=experiment_name,
                        FOLD=fold,
                        LOWERCASE=lowercase,
                        WORDS_PER_SUBDOCUMENT=words_per_subdocument
                    )
                    all_subdocuments.append(subdocuments)
                for feature_selection_method in experiment["variables"].get("FEATURE_SELECTION_METHOD_VALUES", ["stopwords"]):
                    for num_features_to_keep in experiment["variables"].get("NUM_FEATURES_TO_KEEP_VALUES", [100]):
                        representations = env.ExtractRepresentations(
                            "work/${EXPERIMENT_NAME}/${FOLD}/${LOWERCASE}/${WORDS_PER_SUBDOCUMENT}/${FEATURE_SELECTION_METHOD}/${NUM_FEATURES_TO_KEEP}/representations.json",
                            all_subdocuments,
                            EXPERIMENT_NAME=experiment_name,
                            FOLD=fold,
                            LOWERCASE=lowercase,
                            NUM_FEATURES_TO_KEEP=num_features_to_keep,
                            WORDS_PER_SUBDOCUMENT=words_per_subdocument,
                            FEATURE_SELECTION_METHOD="stopwords"
                        )
                        for cluster_count in experiment["variables"].get("CLUSTER_COUNT_VALUES", [10]):
                            clustering = env.ClusterRepresentations(
                                "work/${EXPERIMENT_NAME}/${FOLD}/${LOWERCASE}/${WORDS_PER_SUBDOCUMENT}/${FEATURE_SELECTION_METHOD}/${NUM_FEATURES_TO_KEEP}/${CLUSTER_COUNT}/clustering.json",                            
                                representations,
                                EXPERIMENT_NAME=experiment_name,
                                FOLD=fold,
                                LOWERCASE=lowercase,
                                NUM_FEATURES_TO_KEEP=num_features_to_keep,
                                WORDS_PER_SUBDOCUMENT=words_per_subdocument,
                                FEATURE_SELECTION_METHOD="stopwords",
                                CLUSTER_COUNT=cluster_count
                            )
                            configuration = env.SaveConfiguration(
                                "work/${EXPERIMENT_NAME}/${FOLD}/${LOWERCASE}/${WORDS_PER_SUBDOCUMENT}/${FEATURE_SELECTION_METHOD}/${NUM_FEATURES_TO_KEEP}/${CLUSTER_COUNT}/configuration.json",
                                [],
                                EXPERIMENT_NAME=experiment_name,
                                FOLD=fold,
                                LOWERCASE=lowercase,
                                NUM_FEATURES_TO_KEEP=num_features_to_keep,
                                WORDS_PER_SUBDOCUMENT=words_per_subdocument,
                                FEATURE_SELECTION_METHOD="stopwords",
                                CLUSTER_COUNT=cluster_count
                            )                    
                            results.append((clustering, configuration))
    clustering_summary = env.EvaluateClusterings(
        "work/${EXPERIMENT_NAME}/clustering_summary.png",
        results,
        EXPERIMENT_NAME=experiment_name,
    )

    # Run and evaluate classification, applied both to unmodified and modified documents
    for mod_name, mod_docs in modified_documents.items():
        pass
