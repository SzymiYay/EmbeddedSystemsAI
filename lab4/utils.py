import torch
import tqdm
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):

    @abstractmethod
    def __call__(self, y_pred, y_ref) -> Any:
        raise NotImplementedError()


class AccuracyMetric(BaseMetric):

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(
        self, y_pred: torch.Tensor, y_ref: torch.Tensor
    ) -> torch.Tensor:
        """
        :param y_pred: tensor of shape (batch_size, num_of_classes) type float
        :param y_ref: tensor with shape (batch_size,) and type Long
        :return: scalar tensor with accuracy metric for batch
        """
        predicted_classes = torch.argmax(y_pred, dim=1)
        correct_predictions = (predicted_classes == y_ref).float()
        score: torch.Tensor = correct_predictions.mean()

        return score


def train_one_epoch(
    model, train_loader, loss_fn, metric, optimizer, update_period, device
):
    model.train()
    total_loss_train = 0.0
    total_acc_train = 0.0
    samples_num_train = 0

    with tqdm.tqdm(train_loader, colour="red", ncols=100) as t:
        for i, (X, y) in enumerate(t):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss.backward()

            if (i + 1) % update_period == 0:
                optimizer.step()
                optimizer.zero_grad()

            accuracy = metric(y_pred, y)
            batch_size = y.size(0)
            total_loss_train += loss.item() * batch_size
            total_acc_train += accuracy.item() * batch_size
            samples_num_train += batch_size

            current_loss = total_loss_train / samples_num_train
            current_acc = total_acc_train / samples_num_train

            t.set_postfix(
                loss=f"{current_loss:.4f}", accuracy=f"{current_acc:.4f}"
            )

    epoch_loss_train = total_loss_train / samples_num_train
    epoch_acc_train = total_acc_train / samples_num_train

    return epoch_loss_train, epoch_acc_train


def test_one_epoch(model, test_loader, loss_fn, metric, device):
    model.eval()
    total_loss_test = 0.0
    total_acc_test = 0.0
    samples_num_test = 0

    with torch.no_grad():
        with tqdm.tqdm(test_loader, colour="green", ncols=100) as t:
            for X, y in t:
                X, y = X.to(device), y.to(device)

                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                accuracy = metric(y_pred, y)

                batch_size = y.size(0)
                total_loss_test += loss.item() * batch_size
                total_acc_test += accuracy.item() * batch_size
                samples_num_test += batch_size

                current_loss = total_loss_test / samples_num_test
                current_acc = total_acc_test / samples_num_test

                t.set_postfix(
                    loss=f"{current_loss:.4f}", accuracy=f"{current_acc:.4f}"
                )

    epoch_loss_test = total_loss_test / samples_num_test
    epoch_acc_test = total_acc_test / samples_num_test

    return epoch_loss_test, epoch_acc_test


def test_or_train(
    model,
    train_loader,
    test_loader,
    loss_fn,
    metric,
    optimizer,
    update_period,
    epoch_max,
    device,
    mode="train",
    early_stopping_accuracy=None,
):
    loss_train = []
    loss_test = []
    acc_train = []
    acc_test = []

    for e in range(epoch_max):
        print(f"Epoch: {e + 1}/{epoch_max}")

        if mode in ["train", "both"]:
            epoch_loss_train, epoch_acc_train = train_one_epoch(
                model,
                train_loader,
                loss_fn,
                metric,
                optimizer,
                update_period,
                device,
            )
            loss_train.append(epoch_loss_train)
            acc_train.append(epoch_acc_train)

        if mode in ["test", "both"]:
            epoch_loss_test, epoch_acc_test = test_one_epoch(
                model, test_loader, loss_fn, metric, device
            )
            loss_test.append(epoch_loss_test)
            acc_test.append(epoch_acc_test)

        if (
            early_stopping_accuracy
            and mode in ['train', 'both']
            and epoch_acc_train >= early_stopping_accuracy
        ):
            print(
                f"Training accuracy of {epoch_acc_train:.4f} achieved, stopping training."
            )
            break

    return model, {
        "loss_train": loss_train,
        "acc_train": acc_train,
        "loss_test": loss_test,
        "acc_test": acc_test,
    }


def draw_acc_loss(epochs, history):
    loss_train = history["loss_train"]
    loss_test = history["loss_test"]
    acc_train = history["acc_train"]
    acc_test = history["acc_test"]

    loss_train_shape = len(loss_train)
    loss_test_shape = len(loss_test)
    acc_train_shape = len(acc_train)
    acc_test_shape = len(acc_test)

    if (
        loss_train_shape != loss_test_shape
        or acc_train_shape != acc_test_shape
    ):
        raise ValueError(
            f"Different number of epochs for train and test loss: {loss_train_shape} != {loss_test_shape} or train and test accuracy: {acc_train_shape} != {acc_test_shape}"
        )

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss_train"], label="Train loss")
    plt.plot(epochs, history["loss_test"], label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["acc_train"], label="Train accuracy")
    plt.plot(epochs, history["acc_test"], label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()
