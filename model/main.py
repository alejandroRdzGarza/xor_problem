import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)

#inputs and expected outputs for XOR
inputs = np.array([[0,0],
                   [0,1],
                   [1,0],
                   [1,1]])

expected_output = np.array([[0],
                            [1],
                            [1],
                            [0]])

#seed the random number generator for reproducibility
np.random.seed(42)


#Initialize weights and biases
hidden_weights = np.random.uniform(size=(2,2))
hidden_bias = np.random.uniform(size=(1, 2))
output_weights = np.random.uniform(size=(2, 1))
output_bias = np.random.uniform(size=(1, 1))

#learning rate
learning_rate = 0.1

# Training the neural network
for _ in range(10000):
  #Forward propgation
  hidden_layer_activation = np.dot(inputs, hidden_weights)
  hidden_layer_activation += hidden_bias
  hidden_layer_output = sigmoid(hidden_layer_activation)

  output_layer_activation = np.dot(hidden_layer_output, output_weights)
  output_layer_activation += output_bias
  predicted_output = sigmoid(output_layer_activation)

  #Backpropagation
  error = expected_output - predicted_output
  d_predicted_output = error * sigmoid_derivative(predicted_output)

  error_hidden_layer = d_predicted_output.dot(output_weights.T)
  d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

  #Updating weights and biases
  output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
  output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
  hidden_weights += inputs.T.dot(d_hidden_layer) * learning_rate
  hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
  
#predictions after training
print("Predicted output: \n", predicted_output)