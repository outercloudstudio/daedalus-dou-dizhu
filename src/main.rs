mod connect_four;

use std::cell::RefCell;
use std::io;
use std::rc::Rc;

use connect_four::{ConnectFourGame, ConnectFourModel, ConnectFourState};
use tch::nn::{Optimizer, OptimizerConfig};
use tch::{Device, Kind, Tensor, nn};

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

    for _ in 0..30 {
        mcts_connect_four(node.clone(), game, model, false);
    }

    let result = game.result();

    if result == 0 && node.borrow().moves.as_ref().unwrap().len() > 0 {
        history.push(node.clone());

        let mut best_move: Option<Rc<RefCell<ConnectFourState>>> = None;
        let mut best_score = 0f64;

        for game_move in node.borrow().moves.as_ref().unwrap() {
            let game_move_access = game_move.borrow();

            let score = game_move_access.get_score();

            if best_move.is_none() || score > best_score {
                best_move = Some(game_move.clone());
                best_score = score;
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
    let mut total_loss = 0f64;

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

        let (policy, score) = model.forward(&game.board_state);

        if display {
            println!("Score {}", score.double_value(&[]));
        }

        let target_policy = Tensor::zeros(&[7], (Kind::Float, tch::Device::Cpu));

        for game_move in history_node.borrow().moves.as_ref().unwrap() {
            let _ = target_policy
                .get(game_move.borrow().game_move.unwrap())
                .fill_(game_move.borrow().visits);
        }

        let target_policy = &target_policy / target_policy.sum(Kind::Float);
        let target_score = Tensor::from((result * game.perspective) as f64);

        let loss = model.loss(&policy, &score, &target_policy, &target_score);

        total_loss += loss.double_value(&[]).abs();

        optimizer.zero_grad();
        optimizer.backward_step(&loss);
    }

    for history_node in history.iter() {
        if history_node.borrow().game_move.is_some() {
            game.undo_move();
        }
    }

    println!("Loss {}", total_loss / history.len() as f64);
}

fn main() {
    let mut game = ConnectFourGame::new();

    let mut vs = nn::VarStore::new(Device::Cpu);
    let model = ConnectFourModel::new(&vs.root());

    vs.load("./checkpoints/connect_four_04200.ckpt").unwrap();

    loop {
        game.display();

        let result = game.result();

        if result != 0 || game.valid_moves().len() == 0 {
            println!("Finished game with result {}", result);

            break;
        }

        if game.perspective == 1 {
            let mut input = String::new();

            println!("Enter move>");
            io::stdin().read_line(&mut input).expect("Failed to read line");
            let move_position: i64 = input.trim().parse().expect("Please enter a valid number");

            game.make_move(move_position);
        } else {
            let state = Rc::new(RefCell::new(ConnectFourState::new(None, 0f64)));

            for _ in 0..30 {
                mcts_connect_four(state.clone(), &mut game, &model, false);
            }

            let mut best_move: Option<Rc<RefCell<ConnectFourState>>> = None;
            let mut best_score = 0f64;

            for game_move in state.borrow().moves.as_ref().unwrap() {
                let game_move_access = game_move.borrow();

                let score = game_move_access.get_score();

                if best_move.is_none() || score > best_score {
                    best_move = Some(game_move.clone());
                    best_score = score;
                }
            }

            let (policy, score) = model.forward(&game.board_state);

            println!("Score {}", score.double_value(&[]));

            game.make_move(best_move.as_ref().unwrap().borrow().game_move.unwrap());
        }
    }

    // let mut optimizer = nn::Adam::default().build(&vs, 1e-3).unwrap();

    // for i in 0..100000 {
    //     let state = Rc::new(RefCell::new(ConnectFourState::new(None, 0f64)));
    //     let mut history = Vec::new();

    //     let result = play_game(state, &mut game, &model, &mut history, false);
    //     train(result, &mut game, &model, &mut history, &mut optimizer, i % 10 == 0);

    //     if i % 100 == 0 {
    //         vs.save(format!("./checkpoints/connect_four_{:05}.ckpt", i)).unwrap();
    //     }
    // }
}
