from .cnn import CNN

def build_model(model_config_path):
    return CNN(model_config_path)

