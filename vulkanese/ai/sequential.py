import numpy as np
class Sequential:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        self.metrics = []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, loss, metrics=None):
        # Map string identifiers to actual functions or classes
        optimizers = {
            'adam': AdamOptimizer,
            'sgd': SGDOptimizer
        }
        losses = {
            'mean_squared_error': mean_squared_error,
            'cross_entropy': cross_entropy
        }
        self.optimizer = optimizers[optimizer]()
        self.loss = losses[loss]
        self.metrics = metrics or []

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_on_batch(self, x, y):
        # Forward pass
        predictions = self.forward(x)
        # Compute loss
        loss_value = self.loss(predictions, y)
        # Backward pass
        gradient = self.loss.gradient(predictions, y)
        self.backward(gradient)
        # Update weights
        self.optimizer.step(self.layers)

        # Compute metrics
        results = {'loss': loss_value}
        for metric in self.metrics:
            results[metric.__name__] = metric(predictions, y)
        return results

    def fit(self, x_train, y_train, epochs, batch_size, verbose=True):
        n_samples = len(x_train)
        for epoch in range(epochs):
            # Shuffle the training data (optional but recommended)
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_losses = []
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                x_batch = x_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward pass
                predictions = self.forward(x_batch)
                
                # Compute loss and initial gradient
                loss = self.loss(predictions, y_batch)
                epoch_losses.append(loss)
                grad = self.loss.gradient(predictions, y_batch)
                
                # Backward pass
                self.backward(grad)
                
                # Update weights with the optimizer
                self.optimizer.step(self.layers)
            
            # Aggregate and print epoch results if verbose
            if verbose:
                avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

class AdamOptimizer:
    def step(self, layers):
        # Implement the Adam update rule
        pass

class SGDOptimizer:
    def step(self, layers):
        # Implement the SGD update rule
        pass

def mean_squared_error(predictions, targets):
    return ((predictions - targets) ** 2).mean()

def cross_entropy(predictions, targets):
    # Placeholder for cross-entropy calculation
    pass

# Example metric function
def mean_absolute_error(predictions, targets):
    return abs(predictions - targets).mean()

if __name__ == "__main__":
    # Usage
    model = Sequential()
    model.add(Dense(10, 5))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[mean_absolute_error])
