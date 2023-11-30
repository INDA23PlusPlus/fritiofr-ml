use rand::prelude::*;
use simple_matrix::Matrix;

pub struct NeuralNetwork {
    weights: Vec<Matrix<f64>>,
    biases: Vec<Matrix<f64>>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<usize>) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        let mut rng = rand::thread_rng();

        for (prev, next) in layers
            .iter()
            .take(layers.len() - 1)
            .zip(layers.iter().skip(1))
        {
            weights.push(Matrix::from_iter(
                *next,
                *prev,
                (0..).map(|_| rng.gen::<f64>() - 1.0),
            ));
            biases.push(Matrix::from_iter(
                *next,
                1,
                (0..).map(|_| rng.gen::<f64>() - 1.0),
            ));
        }

        Self { weights, biases }
    }

    pub fn feed_forward(&self, input: &Vec<f64>) -> Vec<f64> {
        let inp = Matrix::from_iter(input.len(), 1, input.clone().into_iter());

        let res = self.feed_forward_matrix(&inp);

        res.into_iter().collect()
    }

    fn feed_forward_matrix(&self, input: &Matrix<f64>) -> Matrix<f64> {
        let mut res = input.clone();

        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            res = w.clone() * res;
            res = res + b.clone();
            res.apply_mut(|x| *x = NeuralNetwork::activation(x));
        }

        res
    }

    fn activation(input: &f64) -> f64 {
        1.0 / (1.0 + (-input).exp())
    }

    fn activation_derivative(input: &f64) -> f64 {
        NeuralNetwork::activation(input) * (1.0 - NeuralNetwork::activation(input))
    }
}
