import time
import numpy as np
from src.utils import (contractive_penalty_v2 as contractive_penalty, decoder_penalty_v2 as decoder_penalty,
                       print_losses_and_time, print_metric)
import torch
from torch.utils.tensorboard import SummaryWriter
# from torchviz import make_dot


class AutoencoderTrainer:

    def __init__(self, autoencoder, optimizer, loss_function, training_loader, validation_loader, device, metric=None,
                 load_model_filename=None, save_model_filename=None, use_contractive_penalty=False,
                 use_decoder_penalty=False, hyperparams=None):
        """
        Initializes an autoencoder trainer object
        :param device:
        :param autoencoder: Autoencoder model to train. (Must inherit from nn.Module and implement the forward method.)
        :param optimizer: The optimizer to use for training.
        :param loss_function: The loss function to use when training
        :param training_loader: DataLoader for training data
        :param validation_loader: DataLoader for validation data
        :param device: torch.device to run on
        :param metric: Metric to monitor during training
        :param load_model_filename: Provide a filename to load model for continued training (Optional)
        :param save_model_filename: Provide a filename to save model (Optional)
        :param use_contractive_penalty: Boolean, whether to use contractive penalty term in training (Optional)
        :param use_decoder_penalty: Boolean, whether to use decoder penalty term in training (Optional)
        :param hyperparams: Dictionary of hyperparameters 'lambda_c' and 'lambda_d' for respective penalty terms.
        Defaults to 1 unless a valid dictionary is provided. (Example data {'lambda_c': 0.1, 'lambda_d': 0.1})
        """

        self.autoencoder = autoencoder
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.metric = metric
        self.writer = SummaryWriter()
        self.best_loss = np.power(10, 64, dtype=np.float64)  # Initialize to large number
        self.losses = []
        self.metric_values = []
        self.save_model_filename = save_model_filename
        self.use_contractive_penalty = use_contractive_penalty
        self.use_decoder_penalty = use_decoder_penalty
        self.device = device

        if hyperparams is not None:
            self.lambda_c = hyperparams['lambda_c']
            self.lambda_d = hyperparams['lambda_d']
        else:
            self.lambda_c = 1.0
            self.lambda_d = 1.0

        if load_model_filename:
            self.load_checkpoint(load_model_filename)

    def train_one_epoch(self):
        # Training mode
        self.autoencoder.train(True)

        running_loss = 0.0
        running_metric = 0.0
        it = 0
        for i, (_, labels) in enumerate(self.training_loader):
            # Reset gradients
            self.optimizer.zero_grad()

            # Get model prediction
            labels = labels.to(self.device)
            output = self.autoencoder(labels)

            # Compute loss and loss gradient
            loss = self.compute_loss(output, labels)

            # Compute metric values
            if self.metric is not None:
                metric = self.metric(output, labels)
                running_metric += metric.item()

            # make_dot(loss).render('autoencoder_cpv2', format='png')

            # Take step
            loss.backward()
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
            epoch_validation_loss, epoch_validation_metric = self.validation_loss()
            end = time.perf_counter()
            elapsed = end - start

            # Store losses
            self.losses.append([epoch, epoch_training_loss, epoch_validation_loss])
            self.metric_values.append([epoch, epoch_training_metric, epoch_validation_metric])

            # Print and write losses and metrics
            if self.metric is not None:
                self.write_summary(epoch, epoch_training_loss, epoch_validation_loss,
                                   epoch_training_metric, epoch_validation_metric)
                if epoch % 10 == 0:
                    print_metric(epoch, epoch_training_metric, epoch_validation_metric, elapsed)
            else:
                self.write_summary(epoch, epoch_training_loss, epoch_validation_loss)
                if epoch % 10 == 0:
                    print_losses_and_time(epoch, epoch_training_loss, epoch_validation_loss, elapsed)

            # Save a checkpoint if there is a new best validation loss for the epoch
            if epoch_validation_loss < self.best_loss and self.save_model_filename:
                self.best_loss = epoch_training_loss
                self.save_checkpoint()

        # Flush and close writer
        self.writer.flush()
        self.writer.close()

    def validation_loss(self):
        # Evaluation mode
        self.autoencoder.eval()
        running_loss = 0.0
        running_metric = 0.0

        # Calculate the average validation loss
        it = 0
        for i, (_, labels) in enumerate(self.validation_loader):
            labels = labels.to(self.device)
            output = self.autoencoder(labels)
            loss = self.compute_loss(output, labels)
            running_loss += loss.item()

            # Compute metric values
            if self.metric is not None:
                metric = self.metric(output, labels)
                running_metric += metric.item()

            it += 1
        epoch_validation_loss = running_loss / it
        epoch_validation_metric = running_metric / it
        return epoch_validation_loss, epoch_validation_metric

    def compute_loss(self, output, labels):
        labels = labels.requires_grad_(True)
        loss = self.loss_function(labels, output)
        if self.use_contractive_penalty:
            loss = loss + self.lambda_c * contractive_penalty(self.autoencoder, labels, self.device).mean()
        if self.use_decoder_penalty:
            loss = loss + self.lambda_d * decoder_penalty(self.autoencoder, labels, self.device).mean()
        return loss

    def write_summary(self, epoch, training_loss, validation_loss, training_metric=None, validation_metric=None):
        # Writes the current training loss and validation loss to tensorboard
        self.writer.add_scalar("Loss/training", training_loss, epoch)
        self.writer.add_scalar("Loss/validation", validation_loss, epoch)

        # Writes metric values also if they are provided
        if training_metric is not None:
            self.writer.add_scalar("Metric/training", training_metric, epoch)
        if validation_metric is not None:
            self.writer.add_scalar("Metric/validation", validation_metric, epoch)

    def save_checkpoint(self):
        state = {
            'model_state_dict': self.autoencoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': np.array(self.losses),
            'best_loss': self.best_loss,
            'metric_values': np.array(self.metric_values)
        }
        torch.save(state, self.save_model_filename)
        print(f"Saved checkpoint to, \"{self.save_model_filename}\"")

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.losses = checkpoint['losses'].tolist()
        self.metric_values = checkpoint['metric_values'].tolist()
        print(f"Loaded model, \"{filename}\"")
