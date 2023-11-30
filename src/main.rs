use nn::neural_network::NeuralNetwork;

fn main() {
    let nn = NeuralNetwork::new(vec![2, 100, 30, 10]);
    let input = vec![0.0, 1.0];
    let output = nn.feed_forward(&input);

    println!("{:?}", output);
}
