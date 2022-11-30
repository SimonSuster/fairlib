import logging
from pathlib import Path

import torch


def print_network(net, verbose=False):
    """print the NN architecture and number of parameters

    Args:
        net (torch.Module): the model object.
        verbose (bool, optional): whether or not print the model architecture. Defaults to False.
    """
    num_params = 0
    for i, param in enumerate(net.parameters()):
        num_params += param.numel()
    if verbose:
        logging.info(net)
    logging.info('Total number of parameters: %d\n' % num_params)


def save_checkpoint(
        epoch, epochs_since_improvement, model, loss, dev_evaluations,
        valid_confusion_matrices, test_confusion_matrices,
        test_evaluations, is_best, checkpoint_dir, prefix="checkpoint",
        dev_predictions=None, dev_logits=None, dev_probs=None, dev_labels=None, test_predictions=None, test_logits=None,
        test_probs=None, test_labels=None, dev_private_labels=None, test_private_labels=None):
    """save check points to a specified file.

    Args:
        epoch (float): number of epoch of the model training.
        epochs_since_improvement (int): epoch since the best epoch is updated.
        model (torch.module): the trained model.     
        loss (float): training loss.
        dev_evaluations (dict): evaluation results over the development set.
        valid_confusion_matrices (dict): a dict of confusion matrices over the validation set.
        test_confusion_matrices (dict): a dict of confusion matrices over the test set.
        test_evaluations (dict): evaluation results over the test set.
        is_best (bool): indicator of whether the current epoch is the best.
        checkpoint_dir (str): path the to checkpoint directory.
        prefix (str, optional): the predict of checkpoint file names. Defaults to "checkpoint".
        dev_predictions (_type_, optional): save the model predictions over the development set if needed. Defaults to None.
        dev_logits (_type_, optional): save the model logits over the dev set if needed. Defaults to None.
        dev_probs (_type_, optional): save the model probs over the dev set if needed. Defaults to None.
        dev_labels (_type_, optional): save the labels over the dev set if needed. Defaults to None.
        test_predictions (_type_, optional): save the model predictions over the test set if needed. Defaults to None.
        test_logits (_type_, optional): save the model logits over the test set if needed. Defaults to None.
        test_probs (_type_, optional): save the model probs over the test set if needed. Defaults to None.
        test_labels (_type_, optional): save the labels over the test set if needed. Defaults to None.
    """

    _state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        # 'model': model.state_dict(),
        'loss': loss,
        # 'dev_predictions': dev_predictions,
        # 'test_predictions': test_predictions,
        "valid_confusion_matrices": valid_confusion_matrices,
        "test_confusion_matrices": test_confusion_matrices,
        'dev_evaluations': dev_evaluations,
        'test_evaluations': test_evaluations
    }

    if dev_predictions is not None:
        _state["dev_predictions"] = dev_predictions
        _state["dev_logits"] = dev_logits
        _state["dev_probs"] = dev_probs
        _state["dev_labels"] = dev_labels
        _state["dev_private_labels"] = dev_private_labels
    if test_predictions is not None:
        _state["test_predictions"] = test_predictions
        _state["test_logits"] = test_logits
        _state["test_probs"] = test_probs
        _state["test_labels"] = test_labels
        _state["test_private_labels"] = test_private_labels

    filename = "{}_epoch{:.2f}".format(prefix, epoch) + '.pth.tar'
    torch.save(_state, Path(checkpoint_dir) / filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        _state["model"] = model.state_dict()
        #_state["test_predictions"] = test_predictions
        #_state["test_logits"] = test_logits
        #_state["test_probs"] = test_probs

        torch.save(_state, Path(checkpoint_dir) / 'BEST_checkpoint.pth.tar')
