import argparse
import os

from src.data.vocab import Vocab
from src.model.seq2seq import Seq2Seq
from src.model.seq2seq_bahdanau import Seq2Seq_Bahdanau
from src.model.seq2seq_luong import Seq2Seq_Luong
from src.task.translation import Translation
from src.utils.logging import setup_logger

def run_assignment_1(args):
    train_path = args.train_path
    val_path   = args.val_path
    test_path  = args.test_path

    logger = setup_logger(
        output=os.path.join(args.log_dir, "assignment_1"),
        name="assignment_1"
    )

    vocab = Vocab(
        path=train_path,
        src_lang="vietnamese",
        tar_lang="english"
    )

    model = Seq2Seq(
        vocab=vocab,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    task = Translation(
        vocab=vocab,
        model=model,
        train_path=train_path,
        dev_path=val_path,
        test_path=test_path,
        logger=logger,
        checkpoint_path=args.checkpoint_dir,
        lr=args.lr,
        batch_size=args.batch_size
    )

    task.train(
        epochs=args.epochs,
        patience=args.patience
    )

    test_loss, test_rouge = task.test()
    print("TEST LOSS:", test_loss)
    print("TEST ROUGE-L:", test_rouge)

def run_assignment_2(args):
    train_path = args.train_path
    val_path   = args.val_path
    test_path  = args.test_path

    logger = setup_logger(
        output=os.path.join(args.log_dir, "assignment_2"),
        name="assignment_2"
    )

    vocab = Vocab(
        path=train_path,
        src_lang="vietnamese",
        tar_lang="english"
    )

    model = Seq2Seq_Bahdanau(
        vocab=vocab,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    task = Translation(
        vocab=vocab,
        model=model,
        train_path=train_path,
        dev_path=val_path,
        test_path=test_path,
        logger=logger,
        checkpoint_path=args.checkpoint_dir,
        lr=args.lr,
        batch_size=args.batch_size
    )

    task.train(
        epochs=args.epochs,
        patience=args.patience
    )

    test_loss, test_rouge = task.test()
    print("TEST LOSS:", test_loss)
    print("TEST ROUGE-L:", test_rouge)

def run_assignment_3(args):
    train_path = args.train_path
    val_path   = args.val_path
    test_path  = args.test_path

    logger = setup_logger(
        output=os.path.join(args.log_dir, "assignment_3"),
        name="assignment_3"
    )

    vocab = Vocab(
        path=train_path,
        src_lang="vietnamese",
        tar_lang="english"
    )

    model = Seq2Seq_Luong(
        vocab=vocab,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    task = Translation(
        vocab=vocab,
        model=model,
        train_path=train_path,
        dev_path=val_path,
        test_path=test_path,
        logger=logger,
        checkpoint_path=args.checkpoint_dir,
        lr=args.lr,
        batch_size=args.batch_size
    )

    task.train(
        epochs=args.epochs,
        patience=args.patience
    )

    test_loss, test_rouge = task.test()
    print("TEST LOSS:", test_loss)
    print("TEST ROUGE-L:", test_rouge)


def main(args):
    if args.assignment == '1':
        run_assignment_1(args)
    elif args.assignment == '2':
        run_assignment_2(args)
    elif args.assignment == '3':
        run_assignment_3(args)
    else:
        raise ValueError(f"Unsupported task: {args.assignment}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--assignment", type=str, default="translation")

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)

    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--checkpoint_path", type=str, default="logs")

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save logs"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints"
    )

    args = parser.parse_args()
    main(args)


# python main.py \
#   --task translation \
#   --train_path src/data/small-PhoMT/small-train.json \
#   --val_path src/data/small-PhoMT/small-dev.json \
#   --test_path src/data/small-PhoMT/small-test.json \
#   --epochs 10 \
#   --batch_size 16


# src/data/small-PhoMT/small-dev.json
# src/data/small-PhoMT/small-test.json
# src/data/small-PhoMT/small-train.json