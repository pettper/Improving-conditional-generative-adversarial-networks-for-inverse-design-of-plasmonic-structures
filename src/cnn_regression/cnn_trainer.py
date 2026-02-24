import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils import print_losses_and_time, print_metric


# Wrapper class for training a model
class ModelTrainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_function,
        training_loader,
        validation_loader,
        device,
        metric=None,
        load_model_filename=None,
        save_model_filename=None,
        has_weighted_loss_function=None,
        write_to_tensorboard=False,
    ):
        """
        Initializes a model trainer object
        :param metric:
        :param model: Model to train. (Must inherit from nn.Module and implement the forward method.)
        :param optimizer: The optimizer to use for training.
        :param loss_function: The loss function to use when training
        :param training_loader: DataLoader for training data
        :param validation_loader: DataLoader for validation data
        :param device: torch.device to run on
        :param load_model_filename: Provide a filename to load model for continued training (Optional)
        :param save_model_filename: Provide a filename to save model (Optional)
        :param has_weighted_loss_function: flag to indicate if loss function is weighted
        """

        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.has_weighted_loss = has_weighted_loss_function
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.writer = SummaryWriter()
        self.save_model_filename = save_model_filename
        self.losses = []
        self.metric_values = []
        self.best_validation_loss = np.inf
        self.device = device
        self.metric = metric
        self.write_to_tensorboard = write_to_tensorboard

        if load_model_filename:
            self.load_checkpoint(load_model_filename)

    def train_one_epoch(self):
        # Training mode
        self.model.train(True)
        running_loss = 0.0
        running_metric = 0.0
        it = 0
        for i, (inp, labels) in enumerate(self.training_loader):
            # Reset gradients
            self.optimizer.zero_grad()

            # Send to device, either a cpu or a gpu
            inp, labels = inp.to(self.device), labels.to(self.device)

            # Get model prediction
            output = self.model(inp)

            # Compute loss and loss gradient
            loss = self.compute_loss(output, labels)
            loss.backward()

            # Compute metric values
            if self.metric is not None:
                metric = self.metric(output, labels)
                running_metric += metric.item()

            # Take step
            self.optimizer.step()

            # Add loss
            running_loss += loss.item()
            it += 1

        # Return average loss during epoch
        mean_loss = running_loss / it
        mean_metric = running_metric / it
        return mean_loss, mean_metric

    def train_model(self, epochs):
        # Train the model for 'epochs' number of epochs
        for epoch in range(epochs):
            # Train one epoch, collect losses and time
            start = time.perf_counter()
            epoch_training_loss, epoch_training_metric = self.train_one_epoch()
            epoch_validation_loss, epoch_validation_metric = (
                self.calculate_validation_loss()
            )
            end = time.perf_counter()
            elapsed = end - start

            # Store losses
            self.losses.append([epoch, epoch_training_loss, epoch_validation_loss])
            self.metric_values.append(
                [epoch, epoch_training_metric, epoch_validation_metric]
            )

            # Print and write losses
            if self.metric is not None:
                self.write_summary(
                    epoch,
                    epoch_training_loss,
                    epoch_validation_loss,
                    epoch_training_metric,
                    epoch_validation_metric,
                )
                if epoch % 10 == 0:
                    print_metric(
                        epoch, epoch_training_metric, epoch_validation_metric, elapsed
                    )
            else:
                self.write_summary(epoch, epoch_training_loss, epoch_validation_loss)
                if epoch % 10 == 0:
                    print_losses_and_time(
                        epoch, epoch_training_loss, epoch_validation_loss, elapsed
                    )

            # Save a checkpoint if there is a new best validation loss for the epoch
            if (
                epoch_validation_loss < self.best_validation_loss
                and self.save_model_filename
            ):
                self.best_validation_loss = epoch_validation_loss
                self.save_checkpoint()

            if epoch % 5 == 0:
                self.write_figure_to_summary("Validation/Sca. cross sec.", epoch)

            # On last epoch, save losses and metrics
            if (epoch == epochs - 1) and self.save_model_filename:
                self.save_on_last_epoch()

        # Flush and close writer
        self.writer.flush()
        self.writer.close()

    def calculate_validation_loss(self):
        # Evaluation mode
        self.model.eval()
        running_loss = 0.0
        running_metric = 0.0

        with torch.no_grad():
            # Calculate the average validation loss
            it = 0
            for i, (inp, labels) in enumerate(self.validation_loader):
                # Send to device, either a cpu or a gpu
                inp, labels = inp.to(self.device), labels.to(self.device)

                # Calculate loss
                output = self.model(inp)
                loss = self.compute_loss(output, labels).item()
                running_loss += loss
                if self.metric is not None:
                    running_metric += self.metric(output, labels).item()
                it += 1
            epoch_validation_loss = running_loss / it
            epoch_validation_metric = running_metric / it
        return epoch_validation_loss, epoch_validation_metric

    def write_summary(
        self,
        epoch,
        training_loss,
        validation_loss,
        training_metric=None,
        validation_metric=None,
    ):
        if self.write_to_tensorboard:
            # Writes the current training loss and validation loss to tensorboard
            self.writer.add_scalar("Loss/training", training_loss, epoch)
            self.writer.add_scalar("Loss/validation", validation_loss, epoch)

            # Writes metric values also if they are provided
            if training_metric is not None:
                self.writer.add_scalar("Metric/training", training_metric, epoch)
            if validation_metric is not None:
                self.writer.add_scalar("Metric/validation", validation_metric, epoch)

    def write_figure_to_summary(self, tag, epoch):
        # Adds a figure with validation example plots
        if self.write_to_tensorboard:
            cpu_device = torch.device("cpu")
            with torch.no_grad():
                fig, ax = plt.subplots(2, 2, figsize=(10, 10))
                inp, labels = next(iter(self.validation_loader))
                inp, labels = inp.to(self.device), labels.to(self.device)
                pred = self.model(inp).to(cpu_device).numpy()
                labels = labels.to(cpu_device).numpy()
                ax[0, 0].plot(pred[0], label="Pred.")
                ax[0, 0].plot(labels[0], label="FEM")
                ax[0, 0].legend()
                ax[0, 1].plot(pred[1], label="Pred.")
                ax[0, 1].plot(labels[1], label="FEM")
                ax[0, 1].legend()
                ax[1, 0].plot(pred[2], label="Pred.")
                ax[1, 0].plot(labels[2], label="FEM")
                ax[1, 0].legend()
                ax[1, 1].plot(pred[3], label="Pred.")
                ax[1, 1].plot(labels[3], label="FEM")
                ax[1, 1].legend()
                plt.suptitle(tag)
                self.writer.add_figure(tag, fig, epoch)

    def save_checkpoint(self):
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": np.array(self.losses),
            "best_validation_loss": self.best_validation_loss,
            "metric_values": np.array(self.metric_values),
        }
        torch.save(state, self.save_model_filename)
        print(f'Saved checkpoint to, "{self.save_model_filename}"')

    def save_on_last_epoch(self):
        # Save losses to best model
        state = torch.load(self.save_model_filename, weights_only=False)
        state["losses"] = np.array(self.losses)
        state["metric_values"] = np.array(self.metric_values)
        torch.save(state, self.save_model_filename)
        # Save full model state on last epoch
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "losses": np.array(self.losses),
            "best_validation_loss": self.best_validation_loss,
            "metric_values": np.array(self.metric_values),
        }
        index_of_last_slash = self.save_model_filename.rfind("/")
        last_epoch_filename = (
            self.save_model_filename[: index_of_last_slash + 1]
            + "last_epoch_"
            + self.save_model_filename[index_of_last_slash + 1 :]
        )
        torch.save(state, last_epoch_filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.losses = checkpoint["losses"].tolist()
        self.metric_values = checkpoint["metric_values"].tolist()
        self.best_validation_loss = checkpoint["best_validation_loss"]
        print(f'Loaded model, "{filename}"')

    def compute_loss(self, output, labels):
        if self.has_weighted_loss:
            weights = self.get_mse_weights(labels)
            loss = self.loss_function(output, labels, weights)
        else:
            loss = self.loss_function(output, labels)
        return loss

    def get_mse_weights(self, labels):
        # MSE Weights are the inverse of the maximum value for each sample
        weights = (1 / labels.abs()).max(dim=2).values.max(dim=1).values
        return weights
