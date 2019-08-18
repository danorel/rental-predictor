import time
import torch
import numpy as np
from sys import float_info
from math import sqrt


def train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        num_epochs,
        metric,
        device,
        print_step
) -> tuple:
    """
    Training the model
    :type model: torch.nn.Module
    :type criterion: torch.nn.MSELoss
    :type optimizer: torch.optim.Optimizer
    :type train_loader: torch.utils.data.DataLoader
    :type num_epochs: int
    :type metric: method
    :type device: torch.device
    :type print_step: int
    """
    loss_values, metric_values, batches_benchmarks = [], [], []
    loss, working_time = 0, 0
    print(f"Starting training the {model.__class__.__name__}")
    for epoch in range(num_epochs):
        prediction, target = 0, 0
        working_time = 0
        for local_batch, local_targets in train_loader:
            stopwatch = time.time()

            data = local_batch \
                .to(device) \
                .type(torch.DoubleTensor)
            target = local_targets \
                .to(device) \
                .type(torch.DoubleTensor)
            prediction = model.forward(data)
            loss = criterion(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            working_time = time.time() - stopwatch
        if epoch != 0 and epoch % print_step == 0:
            print_epoch(
                epoch=epoch,
                loss=loss,
                working_time=working_time,
                criterion=criterion
            )
            if (prediction != prediction).any():
                metric_values.append(
                    calculate_metric_value(
                        metric=metric,
                        prediction=prediction,
                        target=target
                    )
                )
            else:
                metric_values.append(float_info.max)
            loss_values.append(loss.item())
            batches_benchmarks.append(working_time)
    print("Training has been successfully ended!")
    statistics = save_statistics(
        loss_values=loss_values,
        metric_values=metric_values,
        optimizer=optimizer.__class__.__name__,
        metric=metric.__name__,
        num_epochs=num_epochs,
        device=device,
        print_step=print_step,
        working_time=working_time
    )
    return model.parameters(), statistics


def calculate_metric_value(
        metric,
        prediction,
        target
):
    """
    Calculate the metric, converting the tensors to numpy ndarrays
    :type target: torch.tensor
    :type prediction: torch.tensor
    :type metric: method
    """
    prediction = prediction \
        .cpu() \
        .detach() \
        .numpy()
    target = target \
        .cpu() \
        .detach() \
        .numpy()
    return metric(target, prediction)


def print_epoch(
        epoch,
        loss,
        working_time,
        criterion
):
    """
    Print the epoch stats
    :type epoch: int
    :type loss torch.tensor
    :type working_time: float
    :type criterion torch.nn.MSELoss
    """
    print(f'Epoch {epoch}')
    print(f'{criterion.__class__.__name__}: {loss.item()}')
    print('Working out: %.5f micro seconds' % working_time)


def save_statistics(
        loss_values,
        metric_values,
        print_step,
        working_time,
        optimizer=None,
        metric=None,
        num_epochs=None,
        device=None,
) -> dict:
    statistics = {
        'loss_values': loss_values,
        'metric_values': metric_values,
        'working_time': working_time,
        'print_step': print_step
    }
    if optimizer is not None:
        statistics['optimizer'] = optimizer
    if metric is not None:
        statistics['metric'] = metric
    if num_epochs is not None and num_epochs is int:
        statistics['num_epochs'] = num_epochs
    if device is not None:
        statistics['device'] = device
    return statistics


def test_model(
        model,
        loader,
        metric,
        device,
        isValidation=False
):
    """
    Training the model
    :type model: torch.nn.Module
    :type loader: torch.utils.data.DataLoader
    :param metric: method
    :type device: torch.device
    :type isValidation: bool
    """
    if isValidation:
        print('Validating the model!')
    else:
        print('Testing the model!')

    predictions, targets = [], []
    for local_data, local_targets in loader:
        data = local_data \
            .to(device) \
            .type(torch.DoubleTensor)
        target = local_targets \
            .to(device) \
            .type(torch.DoubleTensor)

        prediction = model(data)

        predictions.extend(prediction)
        targets.extend(target)

    predictions = np.array([
        convert_tensor_to_ndarray(prediction) for prediction in predictions
    ])
    targets = np.array([
        convert_tensor_to_ndarray(target) for target in targets
    ])
    return predictions, metric(targets, predictions)


def convert_tensor_to_ndarray(
        tensor
):
    """
    Tensor to ndarray converter
    :type tensor: torch.tensor
    """
    ndarray = tensor \
        .cpu() \
        .detach() \
        .numpy()
    return ndarray
