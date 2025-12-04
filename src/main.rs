mod connect_four;

use rand::Rng;
use std::cell::RefCell;
use std::io;
use std::rc::Rc;

use connect_four::{ConnectFourGame, ConnectFourModel, ConnectFourState};
use tch::nn::{Optimizer, OptimizerConfig, VarStore};
use tch::{Device, Kind, Tensor, nn, vision};

use crate::connect_four::mcts_connect_four;

fn play_game(
    node: Rc<RefCell<ConnectFourState>>,
    game: &mut ConnectFourGame,
    model: &ConnectFourModel,
    history: &mut Vec<Rc<RefCell<ConnectFourState>>>,
    display: bool,
) -> i64 {
    if display {
        game.display();
    }

    for _ in 0..100 {
        mcts_connect_four(node.clone(), game, model, false);
    }

    let result = game.result();

    if result == 0 && node.borrow().moves.as_ref().unwrap().len() > 0 {
        history.push(node.clone());

        let mut best_move: Option<Rc<RefCell<ConnectFourState>>> = None;
        let mut best_visits = 0i64;

        for game_move in node.borrow().moves.as_ref().unwrap() {
            let game_move_access = game_move.borrow();

            let visits = game_move_access.visits;

            if best_move.is_none() || visits > best_visits {
                best_move = Some(game_move.clone());
                best_visits = visits;
            }
        }

        game.make_move(best_move.as_ref().unwrap().borrow().game_move.unwrap());

        let result = play_game(best_move.unwrap().clone(), game, model, history, display);

        game.undo_move();

        return result;
    } else {
        return result;
    }
}

fn train(
    result: i64,
    game: &mut ConnectFourGame,
    model: &ConnectFourModel,
    history: &mut Vec<Rc<RefCell<ConnectFourState>>>,
    optimizer: &mut Optimizer,
    display: bool,
) {
    for history_node in history.iter() {
        if history_node.borrow().game_move.is_some() {
            game.make_move(history_node.borrow().game_move.unwrap());
        }

        if display {
            game.display();

            for game_move in history_node.borrow().moves.as_ref().unwrap() {
                let game_move_access = game_move.borrow();
                let score = game_move_access.get_score();

                if game_move_access.visits > 0 {
                    println!(
                        "Move {} {} {} {} {}",
                        game_move_access.game_move.unwrap(),
                        (score * 100f64).floor() / 100f64,
                        ((game_move_access.score_total as f64) / (game_move_access.visits as f64) * 100f64).floor() / 100f64,
                        game_move_access.visits,
                        (game_move_access.policy * 100f64).floor() / 100f64
                    )
                } else {
                    println!(
                        "Move {} {} no visits {} {}",
                        game_move_access.game_move.unwrap(),
                        (score * 100f64).floor() / 100f64,
                        game_move_access.visits,
                        (game_move_access.policy * 100f64).floor() / 100f64
                    )
                }
            }
        }

        if history_node.borrow().moves.as_ref().unwrap().len() == 0 {
            continue;
        }

        let (policy, score) = model.forward(&game);

        if display {
            println!("Score {}", score.double_value(&[]));
        }

        let target_policy = Tensor::zeros(&[7], (Kind::Float, tch::Device::Cpu));

        for game_move in history_node.borrow().moves.as_ref().unwrap() {
            let _ = target_policy
                .get(game_move.borrow().game_move.unwrap())
                .fill_(game_move.borrow().visits);
        }

        let target_policy = (target_policy.divide(&target_policy.sum(Kind::Float))).to_device(Device::cuda_if_available());
        let target_score = (Tensor::from((result * game.perspective) as f64)).to_device(Device::cuda_if_available());

        let loss = model.loss(&policy, &score, &target_policy, &target_score);

        optimizer.zero_grad();
        optimizer.backward_step(&loss);
    }

    for history_node in history.iter() {
        if history_node.borrow().game_move.is_some() {
            game.undo_move();
        }
    }
}

fn human_vs_model(model: &ConnectFourModel) {
    let mut game = ConnectFourGame::new();

    loop {
        game.display();

        let result = game.result();

        if result != 0 || game.valid_moves().len() == 0 {
            println!("Finished game with result {}", result);

            break;
        }

        if game.perspective == -1 {
            let mut input = String::new();

            println!("Enter move>");
            io::stdin().read_line(&mut input).expect("Failed to read line");
            let move_position: i64 = input.trim().parse().expect("Please enter a valid number");

            game.make_move(move_position);
        } else {
            let state = Rc::new(RefCell::new(ConnectFourState::new(None, 0f64)));

            for _ in 0..100 {
                mcts_connect_four(state.clone(), &mut game, &model, false);
            }

            let mut best_move: Option<Rc<RefCell<ConnectFourState>>> = None;
            let mut best_visits = 0i64;

            for game_move in state.borrow().moves.as_ref().unwrap() {
                let game_move_access = game_move.borrow();

                let score = game_move_access.get_score();
                let visits = game_move_access.visits;

                if best_move.is_none() || visits > best_visits {
                    best_move = Some(game_move.clone());
                    best_visits = visits;
                }

                if game_move_access.visits > 0 {
                    println!(
                        "Move {} {} {} {} {}",
                        game_move_access.game_move.unwrap(),
                        (score * 100f64).floor() / 100f64,
                        ((game_move_access.score_total as f64) / (game_move_access.visits as f64) * 100f64).floor() / 100f64,
                        game_move_access.visits,
                        (game_move_access.policy * 100f64).floor() / 100f64
                    )
                } else {
                    println!(
                        "Move {} {} no visits {} {}",
                        game_move_access.game_move.unwrap(),
                        (score * 100f64).floor() / 100f64,
                        game_move_access.visits,
                        (game_move_access.policy * 100f64).floor() / 100f64
                    )
                }
            }

            let (policy, score) = model.forward(&game);

            println!("Score {}", score.double_value(&[]));

            game.make_move(best_move.as_ref().unwrap().borrow().game_move.unwrap());
        }
    }
}

fn model_vs_model(model_a: &ConnectFourModel, model_b: &ConnectFourModel) {
    let mut game = ConnectFourGame::new();

    loop {
        game.display();

        let result = game.result();

        if result != 0 || game.valid_moves().len() == 0 {
            println!("Finished game with result {}", result);

            break;
        }

        let state = Rc::new(RefCell::new(ConnectFourState::new(None, 0f64)));

        let perspective = game.perspective;

        for _ in 0..100 {
            mcts_connect_four(state.clone(), &mut game, if perspective == 1 { model_a } else { model_b }, false);
        }

        let mut best_move: Option<Rc<RefCell<ConnectFourState>>> = None;
        let mut best_visits = 0i64;

        for game_move in state.borrow().moves.as_ref().unwrap() {
            let game_move_access = game_move.borrow();

            let visits = game_move_access.visits;

            if best_move.is_none() || visits > best_visits {
                best_move = Some(game_move.clone());
                best_visits = visits;
            }
        }

        let (policy, score) = if game.perspective == 1 { model_a } else { model_b }.forward(&game);

        println!("Score {}", score.double_value(&[]));

        game.make_move(best_move.as_ref().unwrap().borrow().game_move.unwrap());
    }
}

struct Participant {
    name: String,
    vs: nn::VarStore,
    model: ConnectFourModel,
    elo: f64,
}

impl Participant {
    fn new(name: String, vs: nn::VarStore, model: ConnectFourModel) -> Participant {
        return Participant {
            name,
            vs,
            model,
            elo: 1500f64,
        };
    }
}

fn main() {
    let mut var_store = nn::VarStore::new(Device::cuda_if_available());
    let model = ConnectFourModel::new(&var_store.root());
    // var_store.load("./checkpoints/connect_four_01000.ckpt").unwrap();

    let mut optimizer = nn::Sgd::default().build(&var_store, 1e-3).unwrap();
    let mut game = ConnectFourGame::new();

    let mut rng = rand::thread_rng();

    for i in 0..3000 {
        let random_number: i64 = rng.random_range(0..7);

        game.make_move(random_number);
        // game.display();

        let (policy, value) = model.forward(&game);

        let target_policy = Tensor::zeros(&[7], (Kind::Float, tch::Device::Cpu));

        let _ = target_policy.get(random_number).fill_(1);

        let target_policy = (target_policy.divide(&target_policy.sum(Kind::Float))).to_device(Device::cuda_if_available());
        let target_value = (Tensor::from(
            (if random_number < 3 {
                1
            } else if random_number > 3 {
                -1
            } else {
                0
            }) as f64,
        ))
        .to_device(Device::cuda_if_available());

        if i % 100 == 0 {
            println!("{}", policy);
            println!("{}", target_policy);
            println!("{}", value);
            println!("{}", target_value);
        }

        let loss = model.loss(&policy, &value, &target_policy, &target_value);

        optimizer.zero_grad();
        optimizer.backward_step(&loss);

        println!("Loss {}", loss.double_value(&[]));

        game.undo_move();
    }

    // for i in 0..3000 {
    //     let state = Rc::new(RefCell::new(ConnectFourState::new(None, 0f64)));
    //     let mut history = Vec::new();

    //     let result = play_game(state, &mut game, &model, &mut history, false);
    //     train(result, &mut game, &model, &mut history, &mut optimizer, i % 10 == 0);

    //     if i % 100 == 0 {
    //         var_store.save(format!("./checkpoints/connect_four_{:05}.ckpt", i)).unwrap();
    //     }
    // }
}
