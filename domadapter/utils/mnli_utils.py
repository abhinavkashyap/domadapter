from datasets import load_dataset
import multiprocessing
import os


def prepare_mnli():
    """Loads and prepares CSVs for Multi-Genre Natural Language Inference.

    Parameters
    ----------

    Returns
    -------
    """
    dataset = load_dataset("multi_nli")

    # remove unnecessary columns
    dataset = dataset.remove_columns(
        [
            "hypothesis_binary_parse",
            "hypothesis_parse",
            "premise_parse",
            "premise_binary_parse",
            "promptID",
        ]
    )

    train_dataset = dataset["train"]
    val_dataset = dataset["validation_matched"]

    genre_unique = set(train_dataset["genre"])

    dataset_cache = os.environ["DATASET_CACHE_DIR"]

    for source in genre_unique:
        for target in genre_unique:
            if source != target:
                source_dataset = train_dataset.filter(
                    lambda example: example["genre"] == source,
                    num_proc=multiprocessing.cpu_count(),
                )
                target_dataset = train_dataset.filter(
                    lambda example: example["genre"] == target,
                    num_proc=multiprocessing.cpu_count(),
                )

                # train dataset taken from train by sampling 90% samples for train
                # validation dataset taken from train by sampling 10% samples for dev

                source_dataset = source_dataset.train_test_split(test_size=0.1)
                target_dataset = target_dataset.train_test_split(test_size=0.1)

                target_dataset_train = target_dataset["train"].remove_columns(
                    ["label"]
                )  # unlabelled train target set

                # test dataset taken from validation_matched by sampling 1945 samples for source, target domains

                source_dataset_test = val_dataset.filter(
                    lambda example: example["genre"] == source,
                    num_proc=multiprocessing.cpu_count(),
                )
                target_dataset_test = val_dataset.filter(
                    lambda example: example["genre"] == target,
                    num_proc=multiprocessing.cpu_count(),
                )

                source_dataset_test = source_dataset_test.select(range(1945))
                target_dataset_test = target_dataset_test.select(range(1945))

                dir_name = os.path.join(dataset_cache, "mnli", f"{source}_{target}")
                os.makedirs(dir_name, exist_ok=True)

                target_dataset_train.to_csv(
                    os.path.join(dir_name, "target_unlabelled.csv"), index=False
                )
                source_dataset["train"].to_csv(
                    os.path.join(dir_name, "train_source.csv"), index=False
                )

                source_dataset["test"].to_csv(
                    os.path.join(dir_name, "dev_source.csv"), index=False
                )
                target_dataset["test"].to_csv(
                    os.path.join(dir_name, "dev_target.csv"), index=False
                )

                source_dataset_test.to_csv(
                    os.path.join(dir_name, "test_source.csv"), index=False
                )
                target_dataset_test.to_csv(
                    os.path.join(dir_name, "test_target.csv"), index=False
                )
