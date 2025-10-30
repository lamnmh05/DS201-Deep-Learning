import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import logging

from lab1.mnist import MNISTDataset, collate_fn
from lab2.lenet import LeNet
from lab2.vinafood21 import VinaFood21, vinafood_collate_fn
from lab2.googlenet import GoogLeNet

# LOGGING CONFIGURATION
logging.basicConfig(
    level=logging.INFO,                                   
    format="%(asctime)s - %(levelname)s - %(message)s",    
    datefmt="%H:%M:%S",                                   
    handlers=[
        logging.StreamHandler(),                          
        logging.FileHandler("lab2/training.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# MODEL CONFIGURATIONS
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using {device} device")

EPOCHS = 10

batch_size = 32
learning_rate = 0.01


mnist_train_image_path = "lab1/mnist/train-images.idx3-ubyte"
mnist_train_label_path = "lab1/mnist/train-labels.idx1-ubyte"
mnist_test_image_path = "lab1/mnist/t10k-images.idx3-ubyte"
mnist_test_label_path = "lab1/mnist/t10k-labels.idx1-ubyte"

mnist_train_dataset = MNISTDataset(mnist_train_image_path, mnist_train_label_path)
mnist_test_dataset = MNISTDataset(mnist_test_image_path, mnist_test_label_path)
mnist_train_dataloader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
mnist_test_dataloader = DataLoader(mnist_test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


vinafood21_train_path = 'lab2/VinaFood21/train'
vinafood21_test_path  = 'lab2/VinaFood21/test'

vinafood21_train_dataset = VinaFood21(vinafood21_train_path)
vinafood21_test_dataset = VinaFood21(vinafood21_test_path)
vinafood21_train_dataloader = DataLoader(vinafood21_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=vinafood_collate_fn)
vinafood21_test_dataloader = DataLoader(vinafood21_test_dataset, batch_size=1, shuffle=False, collate_fn=vinafood_collate_fn)


model_1 = LeNet().to(device)
model_2 = GoogLeNet(num_classes=21).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=learning_rate)
optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=learning_rate)



# EVALUATION FUNCTION
def evaluate(dataloader, model):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        with tqdm(desc=f"Evaluating", unit="it", total=len(dataloader)) as pbar:
            for item in dataloader:
                image, label = item["image"].to(device), item["label"].to(device)
                logits = model(image)
                predicted = logits.argmax(dim=-1).long()

                preds.extend(predicted.cpu().tolist())
                labels.extend(label.cpu().tolist())

                pbar.update()

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# TRAINING LOOP
def train(dataloader, model, loss_fn, optimizer, epochs) -> None:
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for it, item in enumerate(dataloader):
                images, labels = item["image"].to(device), item["label"].to(device)

                # forward
                pred = model(images)
                loss = loss_fn(pred, labels)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                avg_loss = running_loss / (it + 1)

                pbar.set_postfix(loss=f"{avg_loss:.4f}")
                pbar.update()

        logger.info(f"Epoch {epoch+1}/{epochs} finished | Average loss: {avg_loss:.4f}")
        print(f"Epoch {epoch+1}/{epochs} finished | Average loss: {avg_loss:.4f}")


# MAIN FUNCTION
def main():
    # logger.info("Training LeNet model")
    # train(mnist_train_dataloader, model_1, loss_fn, optimizer_1, EPOCHS)
    # metrics_1 = evaluate(mnist_test_dataloader, model_1)
    # logger.info(f"Metrics for LeNet model: {metrics_1}")

    logger.info("Training GoogLeNet model")
    train(vinafood21_train_dataloader, model_2, loss_fn, optimizer_2, EPOCHS)
    metrics_2 = evaluate(vinafood21_test_dataloader, model_2)
    logger.info(f"Metrics for LeNet model: {metrics_2}")

    # for image in vinafood21_train_dataset:
    #     images = image["image"]
    #     print(images.shape)   # <--- check shape here
    #     break
    # for batch in vinafood21_train_dataloader:
    #     images = batch["image"]
    #     print(images.shape)   # <--- check shape here
    #     break

if __name__ == "__main__":
    main()
    