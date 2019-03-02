from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        W1 = np.random.normal(0.0, weight_scale, (input_dim, hidden_dim));
        b1 = np.zeros(hidden_dim);
        W2 = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes));
        b2 = np.zeros(num_classes);
        
        self.params['W1'] = W1;
        self.params['b1'] = b1;
        self.params['W2'] = W2;
        self.params['b2'] = b2;
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        l1_out, l1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1']);
        scores, l2_one_cache = affine_forward(l1_out, self.params['W2'], self.params['b2']);
        # print("scores.shape: ", scores.shape);
        # l2_final_out, l2_final_cache = layers.softmax_loss(l2_one_out, );
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, grads['b2'] = softmax_loss(scores, y);
        loss += self.reg * (0.5 * (np.sum(self.params['W1'] ** 2)) + 0.5 * (np.sum(self.params['W2'] ** 2)));
        # print("loss: ", loss);

        layer_2_grads, grads['W2'], grads['b2'] = affine_backward(grads['b2'], l2_one_cache);
        layer_1_grads, grads['W1'], grads['b1'] = affine_relu_backward(layer_2_grads, l1_cache);
        
        grads['W1'] += self.reg * self.params['W1'];
        grads['W2'] += self.reg * self.params['W2'];
        # grads['b1'] += self.reg * self.params['b1'];
        # grads['b2'] += self.reg * self.params['b2'];

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # prev_W_dim = input_dim;

        # for l_num in range(self.num_layers):
        #     if l_num != (self.num_layers - 1):
        #         next_layer_dim = hidden_dims[l_num];
        #     else:
        #         next_layer_dim = num_classes;
        #     print("W: ", prev_W_dim, next_layer_dim);
        #     self.params['W%d' %(l_num + 1)] = np.random.normal(0.0, weight_scale, (prev_W_dim, next_layer_dim));
        #     self.params['b%d' %(l_num + 1)] = np.zeros(next_layer_dim);
        #     prev_W_dim = next_layer_dim;
        fan_in = input_dim;
        fan_out = hidden_dims[0];

        self.params['W1'] = np.random.normal(0.0, weight_scale, (fan_in, fan_out));
        self.params['b1'] = np.zeros(fan_out);

        for l_num in range(self.num_layers - 1):
            fan_in = hidden_dims[l_num];
            fan_out = hidden_dims[l_num + 1] if l_num < (self.num_layers - 2) else num_classes;
            self.params['W%d' %(l_num + 2)] = np.random.normal(0.0, weight_scale, (fan_in, fan_out));
            self.params['b%d' %(l_num + 2)] = np.zeros(fan_out);
            
        # for l, v in self.params.items():
        #     print("Label: ", l, "\n", "Items: ", v.shape);
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        layer_output = {};
        layer_cache = {};
        fan_in = X;
        fan_out = self.params['W1'];

        # for l_num in range(self.num_layers):
        #     print("X: ", X.shape, " ", "W: ", self.params['W%d' % (l_num + 1)].shape);
        #     layer_output[str(l_num)], layer_cache[str(l_num)] = affine_relu_forward(X, self.params['W%d' % (l_num + 1)], self.params['b%d' % (l_num + 1)]);
        #     fan_in = self.params['W%d' % (l_num + 1)];
        #     fan_out = self.params['W%d' % (l_num + 2)] if l_num < (self.num_layers - 2) else ;
        x_input = X;
        weights = self.params['W1'];
        bias = self.params['b1'];
            
        for l_num in range(self.num_layers):
            weights = self.params['W%d' % (l_num + 1)];
            bias = self.params['b%d' % (l_num + 1)];
            # print("X Input: ", x_input.shape, "weigthsL: ", weights.shape);
            layer_output[str(l_num + 1)], layer_cache[str(l_num + 1)] = affine_relu_forward(x_input, weights, bias);
            x_input = layer_output[str(l_num + 1)];
            

        scores, last_layer_cache = affine_forward(layer_output[str(self.num_layers - 1)], self.params['W%d' % (self.num_layers)], self.params['b%d' % (self.num_layers)]);
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dscores = softmax_loss(scores, y);
        l2_sum = 0;

        grads_wrt_inputs = {};
        
        for l_num in range(self.num_layers):
            l2_sum += (0.5 * (np.sum(self.params['W%d' % (l_num + 1)] ** 2)));
        
        loss += self.reg * l2_sum;
        # print("loss: ", loss);

        last_layer_input_grads, grads['W%d' % (self.num_layers)], grads['b%d' % (self.num_layers)] = affine_backward(dscores, last_layer_cache);
        
        grads_wrt_inputs[str(self.num_layers)] = last_layer_input_grads;

        for l_num in range(self.num_layers-1, 0, -1):
            # print(l_num);
            grads_wrt_inputs[str(l_num)], grads['W%d' % (l_num)], grads['b%d' % (l_num)] = affine_relu_backward(grads_wrt_inputs[str(l_num + 1)], layer_cache[str(l_num)]);
            # grads['W%d' % (l_num)] += self.reg * self.params['W%d' % (l_num)];
        
        for l_num in range(self.num_layers):
            grads['W%d' % (l_num + 1)] += self.reg * self.params['W%d' % (l_num + 1)];    
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
