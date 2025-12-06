mod connect_four;

use rand::{Rng, random};
use std::cell::RefCell;
use std::io;
use std::rc::Rc;

use connect_four::{ConnectFourGame, ConnectFourModel, ConnectFourState};
use tch::nn::{Optimizer, OptimizerConfig, VarStore};
use tch::{Device, Kind, NewAxis, Tensor, nn, vision};

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

    for _ in 0..300 {
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

        let mut target_policy: [f32; 7] = [0f32; 7];

        for game_move in history_node.borrow().moves.as_ref().unwrap() {
            target_policy[game_move.borrow().game_move.unwrap() as usize] = game_move.borrow().visits as f32;
        }

        let target_policy = Tensor::from_slice(&target_policy).to_kind(Kind::Float);

        let target_policy = (target_policy.divide(&target_policy.sum(Kind::Float))).to_device(Device::cuda_if_available());
        let target_score = Tensor::from_slice(&[(result * game.perspective) as f32]).to_device(Device::cuda_if_available());

        if display {
            println!("{}", policy);
            println!("{}", target_policy);
            println!("{}", score);
            println!("{}", target_score);
        }

        let loss = model.loss(&policy, &score, &target_policy, &target_score);

        if display {
            println!("{}", loss);
        }

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

fn model_vs_model(model_a: &ConnectFourModel, model_b: &ConnectFourModel, display: bool, random_start: bool) -> i64 {
    let mut game = ConnectFourGame::new();

    if random_start {
        let mut rng = rand::rng();

        game.make_move(rng.random_range(0..7));
        game.make_move(rng.random_range(0..7));
    }

    loop {
        if display {
            game.display();
        }

        let result = game.result();

        if result != 0 || game.valid_moves().len() == 0 {
            if display {
                println!("Finished game with result {}", result);
            }

            return result;
        }

        let state = Rc::new(RefCell::new(ConnectFourState::new(None, 0f64)));

        let perspective = game.perspective;

        for _ in 0..300 {
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

        if display {
            println!("Score {}", score.double_value(&[]));
        }

        game.make_move(best_move.as_ref().unwrap().borrow().game_move.unwrap());
    }
}

fn model_vs_model_policy(model_a: &ConnectFourModel, model_b: &ConnectFourModel, display: bool, random_start: bool) -> i64 {
    let mut game = ConnectFourGame::new();

    if random_start {
        let mut rng = rand::rng();

        game.make_move(rng.random_range(0..7));
        game.make_move(rng.random_range(0..7));
    }

    loop {
        if display {
            game.display();
        }

        let result = game.result();

        if result != 0 || game.valid_moves().len() == 0 {
            if display {
                println!("Finished game with result {}", result);
            }

            return result;
        }

        let state = Rc::new(RefCell::new(ConnectFourState::new(None, 0f64)));

        let perspective = game.perspective;

        mcts_connect_four(state.clone(), &mut game, if perspective == 1 { model_a } else { model_b }, false);

        let mut best_move: Option<Rc<RefCell<ConnectFourState>>> = None;
        let mut best_policy = 0f64;

        for game_move in state.borrow().moves.as_ref().unwrap() {
            let game_move_access = game_move.borrow();

            let policy = game_move_access.policy;

            if display {
                println!(
                    "Move {} - {}",
                    game_move_access.game_move.unwrap(),
                    (policy * 100f64).floor() / 100f64
                );
            }

            if best_move.is_none() || policy > best_policy {
                best_move = Some(game_move.clone());
                best_policy = policy;
            }
        }

        let (policy, score) = if game.perspective == 1 { model_a } else { model_b }.forward(&game);

        if display {
            println!("Score {}", score.double_value(&[]));
        }

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
    // let mut var_store_04300 = nn::VarStore::new(Device::cuda_if_available());
    // var_store_04300.load("./checkpoints/connect_four_04300.ckpt").unwrap();

    // let model_04300 = ConnectFourModel::new(&var_store_04300.root());

    // let mut var_store_08400 = nn::VarStore::new(Device::cuda_if_available());
    // var_store_08400.load("./checkpoints/connect_four_08400.ckpt").unwrap();

    // let model_08400 = ConnectFourModel::new(&var_store_08400.root());

    // let mut total = 0;

    // for i in 0..1000 {
    //     total += model_vs_model_policy(&model_08400, &model_04300, false, true);
    //     println!("{} {}", total, i)
    // }

    // println!("{}", total as f64 / 1000f64)

    // human_vs_model(&model_08400);

    // let mut participants: Vec<Participant> = Vec::new();

    // for checkpoint in [0, 1500, 2900, 4300] {
    //     let mut var_store = nn::VarStore::new(Device::cuda_if_available());
    //     var_store
    //         .load(format!("./checkpoints/connect_four_{:05}.ckpt", checkpoint))
    //         .unwrap();

    //     let model = ConnectFourModel::new(&var_store.root());

    //     participants.push(Participant::new(format!("Model {:05}", checkpoint), var_store, model));
    // }

    // let mut rng = rand::rng();

    // for i in 0..3000 {
    //     let a = rng.random_range(0..participants.len());

    //     let mut b = rng.random_range(0..(participants.len() - 1));
    //     if b == a {
    //         b = participants.len() - 1
    //     }

    //     let participant_a = &participants[a as usize];
    //     let participant_b = &participants[b as usize];

    //     println!(
    //         "Playing {} ({}) vs {} ({})",
    //         participant_a.name, participant_a.elo, participant_b.name, participant_b.elo
    //     );

    //     let result = model_vs_model(&participant_a.model, &participant_b.model, true, true);

    //     let p1 = 1.0f64 / (1.0f64 + 10.0f64.powf((participant_b.elo - participant_a.elo) / 400.0f64));
    //     let p2 = 1.0f64 / (1.0f64 + 10.0f64.powf((participant_a.elo - participant_b.elo) / 400.0f64));

    //     participants[a as usize].elo += 30.0f64 * (result as f64 / 2.0f64 + 0.5f64 - p1);
    //     participants[b as usize].elo += 30.0f64 * (1.0f64 - (result as f64 / 2.0f64 + 0.5f64) - p2);

    //     println!(
    //         "Result {} - {} ({}) vs {} ({})",
    //         result,
    //         participants[a as usize].name,
    //         participants[a as usize].elo,
    //         participants[b as usize].name,
    //         participants[b as usize].elo
    //     );

    //     if i % 10 == 0 {
    //         println!("Standings:");

    //         participants.sort_by(|a, b| a.elo.partial_cmp(&b.elo).unwrap());

    //         for participant in &participants {
    //             println!("{} ({})", participant.name, participant.elo);
    //         }
    //     }
    // }

    let mut var_store = nn::VarStore::new(Device::cuda_if_available());
    var_store.load("./checkpoints/connect_four_04300.ckpt").unwrap();

    let model = ConnectFourModel::new(&var_store.root());

    let mut optimizer = nn::AdamW::default().build(&var_store, 1e-3).unwrap();
    let mut game = ConnectFourGame::new();

    for i in 4300..30000 {
        println!("Iteration > {}", i);

        let state = Rc::new(RefCell::new(ConnectFourState::new(None, 0f64)));
        let mut history = Vec::new();

        let result = play_game(state, &mut game, &model, &mut history, false);
        train(result, &mut game, &model, &mut history, &mut optimizer, i % 100 == 0);

        if i % 100 == 0 {
            var_store.save(format!("./checkpoints/connect_four_{:05}.ckpt", i)).unwrap();
        }
    }
}
