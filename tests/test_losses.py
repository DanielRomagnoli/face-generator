import torch

from training.losses import get_adversarial_loss


def test_real_prediction_has_low_loss_when_correct():
    criterion = get_adversarial_loss()

    prediction = torch.tensor([0.99])
    target = torch.tensor([1.0])

    loss = criterion(prediction, target)

    assert loss.item() < 0.02


def test_fake_prediction_has_low_loss_when_correct():
    criterion = get_adversarial_loss()

    prediction = torch.tensor([0.01])
    target = torch.tensor([0.0])

    loss = criterion(prediction, target)

    assert loss.item() < 0.02


def test_wrong_prediction_has_higher_loss():
    criterion = get_adversarial_loss()

    correct_prediction = torch.tensor([0.99])
    wrong_prediction = torch.tensor([0.01])

    target = torch.tensor([1.0])

    correct_loss = criterion(
        correct_prediction,
        target,
    )

    wrong_loss = criterion(
        wrong_prediction,
        target,
    )

    assert wrong_loss > correct_loss

    