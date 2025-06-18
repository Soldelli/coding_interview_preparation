# import torch

# # Let's define three tensors: 
# x = torch.tensor(2.0, requires_grad=True)   # required_grad is set to true because we want to track the gradients for this tensor, which will be useful for backpropagation
# w = torch.tensor(3.0, requires_grad=True)
# b = torch.tensor(1.0, requires_grad=True)

# # Let' have a simple forward pass:
# y = x * w + b   # y = 2.0 * 3.0 + 1.0

# # Now we compute the backward pass which computes the gradient of the loss with respect to each parameters involved in the forward pass
# y.backward()    # computes: dy/dx dy/dw dy/db and stores them in the tensor for which the gradient is computed.

# # backward() computes the gradients of the scalar y with respect to all tensors that have requires_grad=True. 
# # It applies the chain rule of calculus to compute the partial derivatives of y with respect to each of the tensors involved in the forward pass.

# # Now we can print these gradients: 
# print(x.grad == 3.0)
# print(w.grad == 2.0)
# print(b.grad == 1.0)

import torch

# Initialize tensors with requires_grad=True for parameters
x = torch.tensor(2.0, requires_grad=False)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# Learning rate
eta = 0.01

# Number of iterations for gradient descent
iterations = 100

for i in range(iterations):
    # Forward pass: compute y
    y = x * w + b  # y = 2.0 * 3.0 + 1.0
    
    # Compute gradients using backward pass
    y.backward()

    # Update parameters using gradient descent
    with torch.no_grad():  # We don't want to track these operations in the computation graph
        # x -= eta * x.grad
        w -= eta * w.grad
        b -= eta * b.grad

        # Zero the gradients after updating
        # x.grad.zero_()
        w.grad.zero_()
        b.grad.zero_()

    # Optionally print the updated values at each step
    if i % 10 == 0:  # Print every 10 iterations
        print(f"Iteration {i}: x={x.item()}, w={w.item()}, b={b.item()}")

# Final values of the parameters after optimization
print(f"Optimized values: x={x.item()}, w={w.item()}, b={b.item()}")