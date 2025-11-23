mod connect_four;

use connect_four::ConnectFourGame;
use connect_four::ConnectFourModel;
use tch::nn::{Module, OptimizerConfig};
use tch::{Device, Kind, Tensor, nn};

fn main() {
    let mut game = ConnectFourGame::new();

    game.display();
    game.make_move(0);
    game.make_move(0);
    game.display();

    let vs = nn::VarStore::new(Device::Cpu);
    let model = ConnectFourModel::new(&vs.root());
    let mut optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();

    let board_state = Tensor::randn(&[6, 7], (Kind::Float, Device::Cpu));

    let target_policy = Tensor::randn(&[7], (Kind::Float, Device::Cpu)).softmax(0, Kind::Float);
    let target_score = Tensor::randn(&[], (Kind::Float, Device::Cpu));

    // for _ in 0..200 {
    //     let (policy, score) = model.forward(&board_state);

    //     println!("Policy: {:?}", policy);
    //     println!("Score: {:?}", score);
    //     println!("Target Policy: {:?}", target_policy);
    //     println!("Target Score: {:?}", target_score);

    //     let loss = model.loss(&policy, &score, &target_policy, &target_score);
    //     println!("Loss: {:?}", loss);

    //     optimizer.backward_step(&loss);
    // }
}
