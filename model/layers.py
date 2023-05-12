from torch import nn


class CONV_2D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.build(args)
    def forward(self, x):
        return self.conv_2d(x)
    def build(self, args):
        self.input_shape, self.out_shape = args[0], args[1]
        ksz = (args[2], args[3])
        stride, paddind = args[4], args[4]
        self.conv_2d = nn.Conv2d(self.input_shape, self.out_shape, ksz, stride, paddind)

def build_conv_2d(args):
    return CONV_2D(args)

def activation(_input, activation_function):
    activation_map= {
        'softmax':nn.functional.softmax,
        'identity':lambda x:x
    }
    return activation_map.get(activation_function,lambda x:x)(_input)

class Classifier(nn.Module):
    def __init__(self, cin,cout, activation):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.activation = activation
        self.build()
    
    def build(self):
        _c = 256
        self.conv = nn.Conv2d(self.cin, _c, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(_c, self.cout)  # to x(b,c2)
    
    def forward(self, x):
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.activation else activation(x,self.activation)

def build_classifier(c_input_shape, num_out, activation_layer_type):
    return Classifier(c_input_shape, num_out, activation_layer_type)
