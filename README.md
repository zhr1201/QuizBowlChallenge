# UMD Computational Linguistics Final Project
Based on https://github.com/Pinafore/qanta-codalab, (read this for how to run the containers), refactored to allow extending the retriever and reranker classes with yaml config files.

## Command line interface changes
Please refer to the qanta/cli.py, train is separeted to train_retriever and train_reranker. And '--config-file' are added for passing the config file of the models.

# For developers
## Branching
Pull your own branch from development and merge from and merge back frequently.
## Adding a retriever or reranker
1. Derive from AbsReranker or AbsRetriever, implement all the abstract methods.
2. Add your class to RETRIEVER_CHOICES or RERANKER_CHOICES in qanta/model_proxy.
3. Write the yaml file for your model and put it under the conf folder.
4. ./cli train_retriever --config-file yaml_file_path or ./cli train_reranker --config-file yaml_file_path to train your model.

TFIDFRetriever is refactored from the original code base as an example.
