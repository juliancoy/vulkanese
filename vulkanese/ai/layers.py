import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim, activation=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        # Initialize weights with a small random value and biases with zeros
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.biases = np.zeros((1, output_dim))
        
    def forward(self, inputs):
        """Perform the forward pass"""
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        if self.activation:
            self.output = self.activation(self.output)
        return self.output

    def backward(self, d_output):
        """Perform the backward pass"""
        # Derivative of the activation function
        if self.activation:
            d_output = d_output * self.activation(self.output, derivative=True)
        
        # Compute gradients
        d_weights = np.dot(self.inputs.T, d_output)
        d_biases = np.sum(d_output, axis=0, keepdims=True)
        d_inputs = np.dot(d_output, self.weights.T)
        
        # Update parameters using gradient descent
        learning_rate = 0.01  # Typically a hyperparameter
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_inputs

    def __repr__(self):
        return f"Dense(layer: {self.input_dim} -> {self.output_dim}, activation={self.activation.__name__ if self.activation else 'None'})"

# Example activation function
def relu(x, derivative=False):
    if derivative:
        return np.where(x > 0, 1, 0)
    return np.maximum(0, x)

# Example usage
layer = Dense(3, 2, activation=relu)
input_data = np.array([[1, 2, 3]])
output = layer.forward(input_data)
print("Output:", output)
print(layer)
