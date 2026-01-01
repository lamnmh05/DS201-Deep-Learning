import argparse

from src.data.vocab import Vocab
from src.models.transformer import TransformerBackbone, TransformerForClassification, TransformerForSeqLabeling
from src.tasks.text_classification import TextClassificationTask
from src.tasks.sequential_labeling import SequentialLabelingTask

def _run_assignment_1():
    train_path = r"src\data\UIT_ViOCD\train_preprocessed.json"
    val_path   = r"src\data\UIT_ViOCD\dev_preprocessed.json"
    test_path  = r"src\data\UIT_ViOCD\test_preprocessed.json"

    vocab = Vocab(train_path, "review", "domain")

    backbone = TransformerBackbone(
        vocab=vocab,
        d_model=256,
        head=8,
        n_layers=4,
        d_ff=1024,
        dropout=0.1
    )
    model = TransformerForClassification(backbone, num_classes=vocab.num_labels)

    task = TextClassificationTask(
        vocab,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        model=model,
        checkpoint_path='checkpoints/assignment_1',
        lr=1e-3
    )

    task.train(epochs=10, patience=5)
    test_loss, test_f1 = task.test()
    print("TEST LOSS:", test_loss)
    print("TEST MACRO-F1:", test_f1)


def _run_assignment_2():
    train_path = r"src\data\PhoNERT\train.json"
    val_path   = r"src\data\PhoNERT\dev.json"
    test_path  = r"src\data\PhoNERT\test.json"

    vocab = Vocab(train_path, "words", "tags")

    backbone = TransformerBackbone(
        vocab=vocab,
        d_model=256,
        head=8,
        n_layers=4,
        d_ff=1024,
        dropout=0.1
    )
    model = TransformerForSeqLabeling(backbone, num_tags=vocab.num_labels)

    task = SequentialLabelingTask(
        vocab,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        model=model,
        checkpoint_path='checkpoints/assignment_2',
        lr=1e-3
    )

    task.train(epochs=10, patience=5)
    test_loss, test_f1 = task.test()
    print("TEST LOSS:", test_loss)
    print("TEST MACRO-F1:", test_f1)


def main(task):
    if task == 1:
        _run_assignment_1()
    elif task == 2:
        _run_assignment_2()
    else:
        raise Exception('Invalid task')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args()
    main(args.task)
