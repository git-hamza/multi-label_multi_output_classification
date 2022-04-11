import time
import numpy as np
import torch
import wandb
import utils.config as cfg
from utils.helper_functions import calculate_metrics, checkpoint_save


if cfg.WANDB_FLAG:
    # please specify the project and entity from your wandb account in order to run the code wandb.
    wandb.init(project="multi_label_classification", entity="analytics")


def train(model, train_loader, val_loader, device):
    """
    This function is used to execute training loop.
    It takes model, train_loader, val_loader and device as an input.
    """
    dataloaders_dict = {"train": train_loader, "val": val_loader}

    # loading optimizer, choose adam as it is one of the best and takes less time to train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # This specific scheduler is used to update the learning rate when val loss stop improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    # loss is set to np.inf so later can be compared with val loss and update it.
    best_loss = np.inf

    # epoch's loop
    for epoch in range(1, cfg.N_EPOCH + 1):
        # in each epoch, training set and validation set will be iterated to compute loss
        for phase in ["train", "val"]:
            start_time = time.time()
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_loss = 0
            accuracy_color = 0
            accuracy_state = 0
            samples = len(dataloaders_dict[phase].dataset)
            # dataloader loop to get batches
            for batch in dataloaders_dict[phase]:
                samples = (len(dataloaders_dict[phase].dataset) // batch["img"].size()[0]) + 1
                optimizer.zero_grad()

                img = batch["img"]
                target_labels = batch["labels"]
                target_labels = {t: target_labels[t].to(device) for t in target_labels}
                with torch.set_grad_enabled(phase == "train"):
                    # computing loss
                    output = model(img.to(device))
                    loss_train, losses_train = model.get_loss(output, target_labels)

                if phase == "train":
                    # backpropagation
                    loss_train.backward()
                    optimizer.step()

                epoch_loss += loss_train.item()
                # computing accuracy per class
                batch_accuracy_color, batch_accuracy_state = calculate_metrics(output, target_labels)
                accuracy_color += batch_accuracy_color
                accuracy_state += batch_accuracy_state

            epoch_loss /= samples
            accuracy_color /= samples
            accuracy_state /= samples

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                # saving checkpoint based on best val loss
                checkpoint_save(model)

            if phase == "val":
                scheduler.step(epoch_loss)

            if cfg.WANDB_FLAG:
                # to save learning curves
                if phase == "val":
                    wandb.log(
                        {
                            "val_loss": epoch_loss,
                            "val_color_accuracy": accuracy_color,
                            "val_state_accuracy": accuracy_state,
                        }
                    )
                else:
                    wandb.log(
                        {
                            "loss": epoch_loss,
                            "train_color_accuracy": accuracy_color,
                            "train_state_accuracy": accuracy_state,
                        }
                    )

            time_taken = time.time() - start_time

            print(
                f"Phase: {phase} | epoch: {epoch} | loss: {epoch_loss:.4f} | color_accuracy: {accuracy_color:.2f} | state_accuracy: {accuracy_state:.2f} | Time: {time_taken:.2f}"
            )
