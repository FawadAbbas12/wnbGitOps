import os
import pytest
import yaml
import csv

@pytest.mark.dependency()
def test_dataset_path():
    pipeine_path = 'data/training.pipeline.yaml'
    assert os.path.exists(pipeine_path), f'unable to find training pipeline file at {pipeine_path}'
    with open(pipeine_path, 'r')as f:
        data = yaml.safe_load(f)['data']
    assert 'train' in data['source'], f'No training data is supplied in config file at {pipeine_path}'
    assert 'val' in data['source'], f'No val data is supplied in config file at {pipeine_path}'
    assert os.path.exists(
        os.path.join(
            data['source']['train'], 'images'
            )
        ), f'unable to find training images path at: {data["source"]["train"]}/images'
    assert os.path.exists(
        os.path.join(
            data['source']['val'], 'images'
            )
        ), f'unable to find training images path at: {data["source"]["val"]}/images'
        
    # assert os.path.exists(data['val']) , f'unable to find val data path at: {data["val"]}'

def validate_dataset_csv_file(file_path, num_heads, classer_per_head):
    with open(file_path) as f:
        dataset_ann = csv.reader(f.readlines())
    csv_rot = file_path.replace(os.path.basename(file_path), '')
    for idx, ann in enumerate(dataset_ann):
        assert os.path.exists(os.path.join(csv_rot, ann[0])), f'Row# {idx}, Missing data in annotation {ann[0]}'
        assert len(ann) > num_heads, f'Row# {idx}, Given annotations are less than number of classification heads {ann}'
        for cw_ann, max_cl in zip(ann[1:], classer_per_head):
            assert int(cw_ann) < max_cl, f"Row# {idx}, Invalid annotaton class id is grater than max classes, Row {ann}"

@pytest.mark.dependency(depends=["test_dataset_path"])
def test_dataset_files():
    pipeine_path = 'data/training.pipeline.yaml'
    with open(pipeine_path, 'r')as f:
        config = yaml.safe_load(f)
    data = config['data']
    num_heads = config['model']['output_heads']
    num_max_classes = config['model']['num_classes']
    classes_per_head= [int(head[1]) if 1 < len(head) else num_max_classes for head in config['model']['classificatoin_head']]
    train_csv = os.path.join(data['source']['train'], 'train.csv') 
    val_csv = os.path.join(data['source']['val'], 'val.csv') 

    assert os.path.exists(train_csv), f'unable to find training labels path at: {train_csv}'
    assert os.path.exists(val_csv), f'unable to find validation labels path at: {val_csv}'
    validate_dataset_csv_file(train_csv, num_heads, classes_per_head)
    validate_dataset_csv_file(val_csv, num_heads, classes_per_head)    
