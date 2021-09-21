from ward import test, fixture, Scope
from domadapter.datamodules.glue_dm import GlueDM
from transformers import AutoTokenizer
from pathlib import Path
import os

TASK_NAMES = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

DATASETS_CACHE_DIR = Path(os.environ["DATASET_CACHE_DIR"]).joinpath(
    "test_dataset_cache"
)


@fixture
def bert_tokenizer(scope=Scope.Module):
    yield AutoTokenizer.from_pretrained("bert-base-uncased")


for task in TASK_NAMES:

    @test("prepare_data for {task_name}", tags=["unit", "data", "slow"])
    def _(task_name=task, tokenizer=bert_tokenizer):
        dm = GlueDM(
            task_name=task_name,
            dataset_cache_dir=DATASETS_CACHE_DIR,
            tokenizer=tokenizer,
        )
        dm.prepare_data()


for task in TASK_NAMES:

    @test("setup_data for {task_name}", tags=["unit", "data", "slow"])
    def _(task_name=task, tokenizer=bert_tokenizer):
        dm = GlueDM(
            task_name=task_name,
            dataset_cache_dir=DATASETS_CACHE_DIR,
            tokenizer=tokenizer,
        )
        dm.prepare_data()
        dm.setup("fit")
        dm.setup("test")
        assert len(dm.train_dataset) > 0
        assert len(dm.val_dataset) > 0
        assert len(dm.test_dataset) > 0


for task in TASK_NAMES:

    @test("data loaders for {task_name}", tags=["unit", "data", "slow"])
    def _(task_name=task, tokenizer=bert_tokenizer):
        dm = GlueDM(
            task_name=task_name,
            dataset_cache_dir=DATASETS_CACHE_DIR,
            tokenizer=tokenizer,
            pad_to_max_length=True,
            max_seq_length=32,
        )
        dm.prepare_data()
        dm.setup("fit")
        train_loader = dm.train_dataloader()
        inputs = next(iter(train_loader))
        attention_mask = inputs["attention_mask"]
        assert attention_mask.size(0) > 0
