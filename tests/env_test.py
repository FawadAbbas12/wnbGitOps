
def test_wandb_version():
    import wandb
    assert wandb.__version__ == '0.15.2', f'Requred WandB version is 0.15.2 but found {wandb.__version__}'


def test_validate_pusher():
    import os
    valid_pushers = os.environ['ALLOWED_PUSHERS']
    pusher = os.environ['PUSHER']
    branch = os.environ['BRANCH']
    assert pusher in valid_pushers.split(','), f'User {pusher} is not allowed to push on {branch} branch'

# print(f'Current WandB version is {wandb.__version__}')

