import torch

from training.losses import get_adversarial_loss


def test_real_prediction_has_low_loss_when_correct():
    criterion = get_adversarial_loss()

    # Large positive logit -> probability close to 1
    prediction = torch.tensor([10.0])
    target = torch.tensor([1.0])

    loss = criterion(prediction, target)

    assert loss.item() < 0.001


def test_fake_prediction_has_low_loss_when_correct():
    criterion = get_adversarial_loss()

    # Large negative logit -> probability close to 0
    prediction = torch.tensor([-10.0])
    target = torch.tensor([0.0])

    loss = criterion(prediction, target)

    assert loss.item() < 0.001


def test_wrong_prediction_has_higher_loss():
    criterion = get_adversarial_loss()

    correct_logit = torch.tensor([10.0])
    wrong_logit = torch.tensor([-10.0])

    target = torch.tensor([1.0])

    correct_loss = criterion(
        correct_logit,
        target,
    )

    wrong_loss = criterion(
        wrong_logit,
        target,
    )

    assert wrong_loss > correct_loss