import numpy as np
import onnx
from onnx import TensorProto
from onnx.helper import (make_model, make_node, make_graph,
                         make_tensor, make_tensor_value_info)
from onnx.checker import check_model
from onnxsim import simplify
import onnxruntime as ort


class Node:
    def __init__(self, node, generation):
        self.generation = generation
        self.node = node

class Graph:
    def __init__(self):
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.initializers = []
        self.last_func_id = None

    def add_node(self, node):
        self.nodes.append(node)

    def set_inputs(self, inputs):
        self.inputs = inputs

    def set_outputs(self, outputs):
        self.outputs = outputs

#===============================================================
#将graph保存为onnx
#===============================================================
def save_graph(graph, model_name, file_name='Example.onnx',ifsimplify=False):
    _graph = make_graph(
        nodes=graph.nodes,
        name=model_name,
        inputs=graph.inputs,
        outputs=graph.outputs,
        initializer=graph.initializers
    )
    onnx_model = make_model(graph=_graph)
    check_model(model=onnx_model)
    if ifsimplify:
        model_simplified, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simplified, file_name)
        print('model saved and simplified successfully')
    else:
        onnx.save_model(onnx_model, file_name)
        print('model saved successfully')


#===============================================================
#载入和使用onnx模型
#===============================================================
def load_model(model_name):
    session = ort.InferenceSession(model_name)
    return session
def model_predict(model, input):
    input=input.astype(np.float32)
    inputs = {"input": input}
    outputs = model.run('output', inputs)
    return outputs

#===============================================================
#创建graph
#===============================================================
def create_graph(output):
    graph = Graph()
    fs = [output.creator]
    graph.last_func_id = id(fs[0])
    graph.outputs.append(make_tensor_value_info('Y', TensorProto.FLOAT, list(output.shape)))
    nodes = []
    while fs:
        f = fs.pop()
        graph_node = _graph_node(f, graph)
        if graph_node.node not in nodes:
            graph.nodes.append(graph_node)
            nodes.append(graph_node.node)
            for input in f.inputs:
                if input.creator is not None and input.creator not in fs:
                    fs.append(input.creator)
    graph.nodes = sorted(graph.nodes, key=lambda x: x.generation)
    graph.nodes = [i.node for i in graph.nodes]
    return graph

#===============================================================
#根据f的名称调用相应的create函数创建节点
#===============================================================
def _graph_node(f, graph):
    name = f.__class__.__name__
    if name in function_nodes:
        return function_nodes[name](f, graph)
    else:
        print(f'Warning:No such function node: {name}')

#===============================================================
#生成输入输出的名称，大多数create函数可用
#===============================================================
def generate_names(f, graph):
    inputs_name = [input.name + f'_{id(input)}' if input.name is not None else f'mid_{id(input)}' for input in f.inputs]
    if f.generation == 0:
        inputs_name[0] = 'X'
        node = make_tensor_value_info('X', TensorProto.FLOAT, list(f.inputs[0].shape))
        graph.inputs.append(node)
    outputs_name = [f'mid_{id(f.outputs[0]())}']
    if id(f) == graph.last_func_id:
        outputs_name = ['Y']
    return inputs_name, outputs_name


#===============================================================
#create函数，创建Function节点
#===============================================================
def create_node(f, graph, node_type, **kwargs):
    inputs_name, outputs_name = generate_names(f, graph)
    node = make_node(
        node_type,
        inputs_name,
        outputs_name,
        name=f'{node_type}_node_{id(f)}',
        **kwargs
    )
    return Node(node, f.generation)

def create_add_node(f, graph):
    return create_node(f, graph, 'Add')
def create_multify_node(f, graph):
    return create_node(f, graph, 'Mul')
def create_relu_node(f, graph):
    return create_node(f, graph, 'Relu')
def create_maxpool_node(f, graph):
    return create_node(f, graph, 'MaxPool', kernel_shape=[f.pool_size, f.pool_size],
                       strides=[f.stride, f.stride], pads=[0, 0, f.pad, f.pad])
def create_AveragePool_node(f, graph):
    return create_node(f, graph, 'AveragePool', kernel_shape=[f.pool_size, f.pool_size],
                       strides=[f.stride, f.stride], pads=[0, 0, f.pad, f.pad])
def create_dropout_node(f, graph):
    return create_node(f, graph, 'Dropout', seed=f.ratio, training_mode=f.training, mask=f.mask)

def create_conv_node(f, graph):
    '''需要初始化权重'''
    inputs_name, outputs_name = generate_names(f, graph)
    graph.initializers.append(make_tensor(f'W_{id(f.inputs[1])}', TensorProto.FLOAT, list(f.inputs[1].shape), f.inputs[1].data))
    try:
        graph.initializers.append(make_tensor(f'b_{id(f.inputs[2])}', TensorProto.FLOAT, list(f.inputs[2].shape), f.inputs[2].data))
    except (AttributeError, IndexError):
        pass
    node = make_node(
        'Conv',
        inputs_name,
        outputs_name,
        kernel_shape=[f.inputs[1].shape[2], f.inputs[1].shape[3]],
        strides=[f.stride, f.stride],
        pads=[f.pad, f.pad, f.pad, f.pad],
        name=f'Conv_node_{id(f)}'
    )
    return Node(node, f.generation)

def create_convtranspose_node(f, graph):
    inputs_name, outputs_name = generate_names(f, graph)
    graph.initializers.append(make_tensor(f'W_{id(f.inputs[1])}', TensorProto.FLOAT, list(f.inputs[1].shape), f.inputs[1].data))
    try:
        graph.initializers.append(make_tensor(f'b_{id(f.inputs[2])}', TensorProto.FLOAT, list(f.inputs[2].shape), f.inputs[2].data))
    except (AttributeError, IndexError):
        pass
    node = make_node(
        'ConvTranspose',
        inputs_name[:-1],
        outputs_name,
        kernel_shape=[f.inputs[1].shape[2], f.inputs[1].shape[3]],
        strides=[f.stride, f.stride],
        pads=[f.pad, f.pad, f.pad, f.pad],
        output_shape=list(f.outputs[0]().shape),
        name=f'ConvTranspose_node_{id(f)}'
    )
    return Node(node, f.generation)

def create_softmaxcrossentropyloss_node(f, graph):
    inputs_name = [f'mid_{id(f.inputs[0])}', 'T']
    graph.inputs.append(make_tensor_value_info('T', TensorProto.INT64, list(f.inputs[1].shape)))
    f.inputs[1].name = 'T'
    outputs_name = 'Y'
    node = make_node(
        'SoftmaxCrossEntropyLoss',
        inputs_name,
        outputs_name,
        reduction='mean',
        ignore_index=None,
        name=f'SoftmaxCrossEntropyLoss_node_{id(f)}'
    )
    return Node(node, f.generation)

def create_batchnormlization_node(f, graph):
    inputs_name, outputs_name = generate_names(f, graph)
    graph.initializers.append(make_tensor(f'scale_{id(f.gamma)}', TensorProto.FLOAT,list(f.gamma.shape),f.gamma))
    graph.initializers.append(make_tensor(f'B_{id(f.beta)}', TensorProto.FLOAT,list(f.beta.shape),f.beta))
    graph.initializers.append(make_tensor(f'input_mean_{id(f.test_mean)}', TensorProto.FLOAT, list(f.test_mean.shape), f.test_mean))
    graph.initializers.append(make_tensor(f'input_var_{id(f.test_var)}', TensorProto.FLOAT, list(f.test_var.shape), f.test_var))
    inputs_name.append(f'scale_{id(f.gamma)}')
    inputs_name.append(f'B_{id(f.beta)}')
    inputs_name.append(f'input_mean_{id(f.test_mean)}')
    inputs_name.append(f'input_var_{id(f.test_var)}')
    if f.training:
        graph.outputs.append(make_tensor_value_info(f'running_mean_{f.generation}', TensorProto.FLOAT, list(f.test_mean.shape)))
        graph.outputs.append(make_tensor_value_info(f'running_var_{f.generation}', TensorProto.FLOAT, list(f.test_var.shape)))
        outputs_name.append(f'running_mean_{f.generation}')
        outputs_name.append(f'running_var_{f.generation}')
    node = make_node(
        'BatchNormalization',
        inputs_name,
        outputs_name,
        epsilon=1e-5,
        momentum=f.momentum,
        training_mode=f.training,
        name=f'BatchNormalization_node_{id(f)}'
    )
    return Node(node, f.generation)
def create_sigmoid_node(f, graph):
    pass
def create_softmax_node(f, graph):
    pass
def create_meansquarederror_node(f, graph):
    pass
def create_Affine_node(f, graph):
    pass
def create_Concat_node(f, graph):
    return create_node(f, graph, 'Concat',axis=f.axis)
#使用节点的字典
function_nodes = {
    'SoftmaxCrossEntropyLoss': create_softmaxcrossentropyloss_node,
    'Sigmoid': create_sigmoid_node,
    'Relu': create_relu_node,
    'Softmax': create_softmax_node,
    'MeanSquaredError': create_meansquarederror_node,
    'Affine': create_Affine_node,
    'BatchNormalization': create_batchnormlization_node,
    'Dropout': create_dropout_node,
    'Conv': create_conv_node,
    'ConvTranspose': create_convtranspose_node,
    'MaxPool': create_maxpool_node,
    'Add': create_add_node,
    'Mul': create_multify_node,
    "Concat": create_Concat_node,
    "AveragePool":create_AveragePool_node
    }