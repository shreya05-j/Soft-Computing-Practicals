# Task 1: To understand and implement the basic working of an artificial neuron

def artificial_neuron(input_signal):
    # A simple artificial neuron with a threshold activation function
    threshold = 0.5
    if input_signal >= threshold:
        return 1
    else:
        return 0

if __name__ == "__main__":
    signal = 0.8
    output = artificial_neuron(signal)
    print("--- Task 1: Basic Artificial Neuron ---")
    print(f"Input signal: {signal}, Neuron Output: {output}")
