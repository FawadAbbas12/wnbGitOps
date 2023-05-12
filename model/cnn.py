import yaml
from torch import nn
from .layers import build_conv_2d, build_classifier

class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)['model']
        elif isinstance(config, dict):
            self.config = config

        self.input_shape = self.config['input_shape']
        self.output_heads = self.config['output_heads']
        self.num_classes = self.config['num_classes']
        bb_config = self.config['backbone']
        classif_head_config = self.config['classificatoin_head']
        self.build_bb(bb_config)
        self.build_classification_head(classif_head_config)

    def build_bb(self, config):
        self.bb_layers = {}
        layer_builder= {
            "conv": build_conv_2d
        }
        for layer_config in config:
            layer_type = layer_config[0]
            layer_args = layer_config[1]
            layer_name = layer_config[2]
            self.bb_layers[layer_name] = layer_builder[layer_type](layer_args)

    def build_classification_head(self, config):
        self.classification_heads = []
        self.classification_head_connection_layers = []
        for classifier_config in config:
            connection_layer_name = classifier_config[0]
            num_out = classifier_config[1] if 1< len(classifier_config) else self.num_classes
            activation_layer_type = classifier_config[2] if 2< len(classifier_config) else False
            c_input_shape = self.bb_layers[connection_layer_name].out_shape
            head = build_classifier(c_input_shape, num_out, activation_layer_type)
            self.classification_heads.append(head)
            self.classification_head_connection_layers.append(connection_layer_name)

    def forward(self, x):
        # classifier_inputs = [v(x) for k,v in self.bb_layers.items() if k in self.classification_head_connection_layers ]
        classifier_inputs = []
        for k,v in self.bb_layers.items():
            x = v(x)
            if k in self.classification_head_connection_layers:
                classifier_inputs.append(x)
        c_out = [ c_head(cin) for cin, c_head in zip(classifier_inputs,self.classification_heads)]
        return c_out
