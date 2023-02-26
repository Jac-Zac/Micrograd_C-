//  nn.hppT
//  Micrograd_C++
//
//  Created by Jacopo Zacchigna on 2023-02-19
//  Copyright © 2023 Jacopo Zacchigna. All rights reserved.

#pragma once
#include "engine.hpp"
#include <random>

template <typename T> T random_uniform(T range_from, T range_to) {
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());
    std::uniform_real_distribution<T> distr(range_from, range_to);
    return distr(generator);
}

// ---------------------------------------------------------

template <typename T> class Neuron {
public:
    Neuron(size_t num_neurons_input);

    // Call operator: w * x + b dot product
    Value<T> operator()(std::vector<Value<T>> &x);

protected:
    // weighted_sum already initialized with the bias
    Value<T> weighted_sum = Value<T>(random_uniform(-1.0, 1.0));
    size_t m_num_neurons_input;
    // Save the return value so that I can draw the graph
    std::vector<Value<T>> m_weights;
};

// ---------------------------------------------------------

template <typename T> class Layer {
public:
    Layer(size_t num_neurons_input, size_t num_neurons_out);

    // Call operator: w * x + b dot product
    std::vector<Value<T>> operator()(std::vector<Value<T>> &x);

protected:
    std::vector<Neuron<T>> m_neurons;
    // Create an array of neurons to return
    std::vector<Value<T>> m_neurons_output;
};

// ----------------------------------------------------------

template <typename T, size_t N> class MLP {
public:
    MLP(size_t num_neurons_input, std::array<size_t, N> num_neurons_out);

    // Call operator: w * x + b dot product
    std::vector<Value<T>> operator()(std::vector<Value<T>> &x);

protected:
    std::vector<Layer<T>> m_layers;
};

//  ================ Implementation  Neuron =================

template <typename T>
Neuron<T>::Neuron(size_t number_of_neurons_input)
    : m_num_neurons_input(number_of_neurons_input) {
    for (size_t i = 0; i < m_num_neurons_input; i++) {
        m_weights.emplace_back(random_uniform(-1.0, 1.0));
    }
}

template <typename T> Value<T> Neuron<T>::operator()(std::vector<Value<T>> &x) {
    // Sum over all multiplies
    for (size_t i = 0; i < m_num_neurons_input; i++) {
        weighted_sum += m_weights[i] * x[i];
    }
    // Return the activation value of the neuron as a value object
    return (weighted_sum.tanh());
}

//  ================ Implementation  Layer =================

template <typename T>
Layer<T>::Layer(size_t num_neurons_input, size_t num_neurons_output) {
    for (size_t i = 0; i < num_neurons_output; i++) {
        m_neurons.emplace_back(Neuron<T>(num_neurons_input));
    }
}

template <typename T>
std::vector<Value<T>> Layer<T>::operator()(std::vector<Value<T>> &x) {

    // Iterate over the m_neurons
    for (auto &neuron : m_neurons) {
        m_neurons_output.emplace_back(neuron(x));
    }
    return m_neurons_output;
}

//  ================ Implementation MLP =================

template <typename T, size_t N>
MLP<T, N>::MLP(size_t num_neurons_input,
               std::array<size_t, N> num_neurons_output) {

    // Create the first layer with the input neuron size
    m_layers.emplace_back(Layer<T>(num_neurons_input, num_neurons_output[0]));

    // Create the following layers
    for (size_t i = 1; i < N; i++) {
        // Create layers with
        m_layers.emplace_back(
            Layer<T>(num_neurons_output[i - 1], num_neurons_output[i]));
    }
}

template <typename T, size_t N>
std::vector<Value<T>> MLP<T, N>::operator()(std::vector<Value<T>> &x) {
    // Iterate over the layers and call sequentially
    for (auto &layer : m_layers) {
        x = layer(x);
    }
    return x;
}
