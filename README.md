# tensorchainer
A lib whose framework learned from keras and chainer for more convenient personal DL&amp;ML developing purpose in TensorFlow

# Requirements
 python 3.+
 tensorflow 1.+.+
 opencv-python, numpy

# General Introduction
 This lib has similar structure as keras does, but not powerful and rich as keras, it's more like a low
 configuration "keras" which extremely constraint with my personal developing purposes, besides, the core
 components are not all keras-like style, they have learned from some features of Chainer and Pytorch (e.g.
 hook mechanism around layer and trainer, a MutableSequence-style LayerList as Chainer does, Pytorch-way
 to control trainable flag...), one thing I must mention is the argument scope in slim also merged into
 this lib, and it has been modified to compatible with function and class in the same time.
 
 Topological graph the most important part in keras also has realized in this lib, with a simpler graph
 algorithm, only convert layers into topological nodes, this is slightly different from keras's.
 
 I didn't follow the way keras does which merges model and trainer together, for my personal developing
 purposes, I made a trainer which has similar structure and training&evaluation procedure as keras's, but
 it's builds and compiles like tensorflow's estimator, due to keras's execution mechanism, it can train
 and evaluate in the same computation graph, unlike estimator, which needs two computation graphs, one for
 training, one for evaluation. This can benifit lots when model's evaluation procedure is same as training.
 
 We have realized some data augments both in opencv-way and tensorflow-way, these augments can be
 serialized into file and deserialize.
 There still have lots works haven't finished inside this lib, but it already enough to support my
 personal developing, Couples of projects have been build base on this lib, and it has made my developing
 procedure more fast and efficient.
