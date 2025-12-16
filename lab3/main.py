import argparse

from data.vocab import Vocab
from model.lstm import LSTM
from model.gru import GRU
from model.bi_lstm import BiLSTM
from tasks.text_classification import TextClassificationTask
from tasks.sequential_labeling import SequentialLabelingTask

def _run_assignment_1():
    train_path = "UIT-VSFC/UIT-VSFC-train.json"
    val_path   = "UIT-VSFC/UIT-VSFC-dev.json"
    test_path  = "UIT-VSFC/UIT-VSFC-test.json"

    vocab = Vocab(train_path, "sentence", "topic")
    model = LSTM(vocab)

    task = TextClassificationTask(
        vocab,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        model=model,
        checkpoint_path= 'src/checkpoint/assignment_1',
        lr=1e-3
    )

    task.train(epochs=20, patience=5)

    test_loss, test_f1 = task.test()
    print("TEST LOSS:", test_loss)
    print("TEST MACRO-F1:", test_f1)

def _run_assignment_2():
    train_path = "UIT-VSFC/UIT-VSFC-train.json"
    val_path   = "UIT-VSFC/UIT-VSFC-dev.json"
    test_path  = "UIT-VSFC/UIT-VSFC-test.json"

    vocab = Vocab(train_path, "sentence", "sentiment")
    model = GRU(vocab)

    task = TextClassificationTask(
        vocab,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        model=model,
        checkpoint_path= 'checkpoint/assignment_2',
        lr=1e-3
    )

    task.train(epochs=20, patience=5)

    test_loss, test_f1 = task.test()
    print("TEST LOSS:", test_loss)
    print("TEST MACRO-F1:", test_f1)

def _run_assignment_3():
    train_path = "PhoNER/train.json"
    val_path   = "PhoNER/dev.json"
    test_path  = "PhoNER/test.json"

    vocab = Vocab(train_path, "words", "tags")
    model = BiLSTM(vocab)

    task = SequentialLabelingTask(
        vocab,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        model=model,
        checkpoint_path= 'checkpoint/assignment_3',
        lr=1e-3
    )

    task.train(epochs=20, patience=5)

    test_loss, test_f1 = task.test()
    print("TEST LOSS:", test_loss)
    print("TEST MACRO-F1:", test_f1)


def main(task):
    if task == 1:
        _run_assignment_1()
    elif task == 2:
        _run_assignment_2()
    elif task == 3:
        _run_assignment_3()
    else:
        raise Exception('Invalid task')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args()
    main(args.task)
