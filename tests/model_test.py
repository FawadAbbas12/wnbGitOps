import os
import pytest
import yaml


@pytest.mark.dependency()
def test_model_config():
    model_file = 'data/training.pipeline.yaml'
    assert os.path.exists(model_file)
    with open(model_file) as f:
        model_config = yaml.safe_load(f)['model']
    assert 'input_shape' in model_config, "Model input shape is not defined in config"
    assert 'output_heads' in model_config, "Nodel dose not define output heads"
    assert 'num_classes' in model_config, "No defaul number of classes provided"
    assert 'backbone' in model_config, "No Backbone config is defiend"
    assert 'classificatoin_head' in model_config, "No classification head is defined in config"
    assert model_config['output_heads'] == len(model_config['classificatoin_head']), f"Model sets {model_config['output_heads']} as number of classificaton heads but {len(model_config['classificatoin_head'])} are defined"
    bb_layers = [layer[-1] for layer in model_config['backbone']]
    for c_head in model_config['classificatoin_head']:
        assert c_head[0] in bb_layers, f"Unable to find {c_head[0]} layer/block in backbone config"


@pytest.mark.dependency(depends=['test_model_config'])
def test_model_infer():
    from model import build_model
    import torch
    model_file = 'data/training.pipeline.yaml'
    with open(model_file) as f:
        model_config = yaml.safe_load(f)['model']
    w,h,c = model_config['input_shape']
    default_classes = model_config['num_classes']
    output_shapes = [c_head[1] if 1<len(c_head) else default_classes for c_head in model_config['classificatoin_head']]
    model = build_model(model_file)
    x = torch.zeros(1, c, w, h, dtype=torch.float, requires_grad=False)
    outputs = model(x)
    for idx, data in enumerate(zip(outputs, output_shapes)):
        output, rqeured_shape = data
        assert output.shape[1] == rqeured_shape, f"Model classificatoin head# {idx} have requred output shape of {rqeured_shape} but got {output.shape}"
