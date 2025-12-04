use std::cell::RefCell;
use std::rc::Rc;

use tch::nn::{Linear, Module, OptimizerConfig, Path};
use tch::{Device, Kind, Tensor, nn};

pub struct ConnectFourGame {
    pub board_state: [i64; 6 * 7],
    pub perspective: i64,
    pub history: Vec<i64>,
}

impl ConnectFourGame {
    pub fn new() -> Self {
        ConnectFourGame {
            board_state: [0; 6 * 7],
            perspective: 1,
            history: Vec::new(),
        }
    }

    pub fn position_to_index(&self, x: i64, y: i64) -> usize {
        return (y * 7 + x) as usize;
    }

    pub fn move_valid(&self, position: i64) -> bool {
        return self.board_state[self.position_to_index(position, 5)] == 0;
    }

    pub fn make_move(&mut self, position: i64) {
        for row in 0..6 {
            if self.board_state[self.position_to_index(position, row)] == 0 {
                self.board_state[self.position_to_index(position, row)] = self.perspective;

                break;
            }
        }

        self.history.push(position);

        self.perspective *= -1;
    }

    pub fn undo_move(&mut self) {
        let last_move = self.history.pop().expect("There was no move left to undo!");

        for i in 0..6 {
            if self.board_state[self.position_to_index(last_move, 5 - i)] != 0 {
                self.board_state[self.position_to_index(last_move, 5 - i)] = 0;

                break;
            }
        }

        self.perspective *= -1;
    }

    pub fn valid_moves(&self) -> Vec<i64> {
        let mut moves = Vec::new();

        for x in 0..7 {
            if self.move_valid(x) {
                moves.push(x);
            }
        }

        moves
    }

    pub fn result(&self) -> i64 {
        for start_y in 0..6 {
            for start_x in 0..4 {
                let looking_for = self.board_state[self.position_to_index(start_x, start_y)];

                if looking_for == 0 {
                    continue;
                }

                let mut found = true;

                for i in 0..3 {
                    if self.board_state[self.position_to_index(start_x + 1 + i, start_y)] != looking_for {
                        found = false;
                        break;
                    }
                }

                if found {
                    return looking_for;
                }
            }
        }

        for start_y in 0..3 {
            for start_x in 0..7 {
                let looking_for = self.board_state[self.position_to_index(start_x, start_y)];

                if looking_for == 0 {
                    continue;
                }

                let mut found = true;

                for i in 0..3 {
                    if self.board_state[self.position_to_index(start_x, start_y + 1 + i)] != looking_for {
                        found = false;
                        break;
                    }
                }

                if found {
                    return looking_for;
                }
            }
        }

        for start_y in 0..3 {
            for start_x in 0..4 {
                let looking_for = self.board_state[self.position_to_index(start_x, start_y)];

                if looking_for != 0 {
                    let mut found = true;

                    for i in 0..3 {
                        if self.board_state[self.position_to_index(start_x + 1 + i, start_y + 1 + i)] != looking_for {
                            found = false;
                            break;
                        }
                    }

                    if found {
                        return looking_for;
                    }
                }

                let looking_for = self.board_state[self.position_to_index(start_x, start_y + 3)];

                if looking_for != 0 {
                    let mut found = true;

                    for i in 0..3 {
                        if self.board_state[self.position_to_index(start_x + 1 + i, start_y + 2 - i)] != looking_for {
                            found = false;
                            break;
                        }
                    }

                    if found {
                        return looking_for;
                    }
                }
            }
        }

        return 0;
    }

    pub fn display(&self) {
        println!("-------");

        for i in 0..6 {
            let mut line = String::new();

            for x in 0..7 {
                let value = self.board_state[self.position_to_index(x, 5 - i)];

                if value == 0 {
                    line.push(' ');
                } else if value == 1 {
                    line.push('O');
                } else if value == -1 {
                    line.push('X');
                }
            }

            println!("{}", line);
        }

        println!("-------");
    }
}

pub struct ConnectFourModel {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
    layer4: Linear,
}

impl ConnectFourModel {
    pub fn new(vs: &Path) -> Self {
        let layer1 = nn::linear(vs / "layer1", 6 * 7, 200, Default::default());
        let layer2 = nn::linear(vs / "layer2", 200, 200, Default::default());
        let layer3 = nn::linear(vs / "layer3", 200, 200, Default::default());
        let layer4 = nn::linear(vs / "layer4", 200, 7 + 1, Default::default());

        ConnectFourModel {
            layer1,
            layer2,
            layer3,
            layer4,
        }
    }

    pub fn forward(&self, game: &ConnectFourGame) -> (Tensor, Tensor) {
        // Flatten and convert to float
        let mut value = (Tensor::from_slice(&game.board_state) * game.perspective)
            .to_kind(Kind::Float)
            .set_requires_grad(true)
            .to_device(Device::cuda_if_available());

        // Forward pass through layers with ReLU activations
        value = value.apply(&self.layer1).relu();
        value = value.apply(&self.layer2).relu();
        value = value.apply(&self.layer3).relu();
        value = value.apply(&self.layer4);

        // Split into policy (first 7) and score (last 1)
        let policy = value.narrow(0, 0, 7);
        let score = value.narrow(0, 7, 1);

        // Apply softmax to policy and tanh to score
        let policy = policy.softmax(0, Kind::Float);
        let score = score.tanh();

        (policy, score)
    }

    pub fn loss(&self, policy: &Tensor, score: &Tensor, target_policy: &Tensor, target_score: &Tensor) -> Tensor {
        let policy_loss = -(target_policy * policy.log()).sum(Kind::Float);

        let value_loss = (score - target_score).pow_tensor_scalar(2);

        policy_loss + value_loss
    }
}

pub struct ConnectFourState {
    pub game_move: Option<i64>,
    pub parent: Option<Rc<RefCell<ConnectFourState>>>,

    pub visits: i64,
    pub score_total: i64,
    pub policy: f64,

    pub moves: Option<Vec<Rc<RefCell<ConnectFourState>>>>,
}

impl ConnectFourState {
    pub fn new(parent: Option<Rc<RefCell<ConnectFourState>>>, policy: f64) -> ConnectFourState {
        return ConnectFourState {
            game_move: None,
            parent: parent,
            visits: 0,
            score_total: 0,
            policy: policy,
            moves: None,
        };
    }

    pub fn get_score(&self) -> f64 {
        let exploration = 0.5f64;

        let parent = self.parent.as_ref().unwrap().borrow();

        if self.visits == 0 {
            return exploration * self.policy * (parent.visits as f64).sqrt() / (1f64 + self.visits as f64);
        }

        return self.score_total as f64 / self.visits as f64
            + exploration * self.policy * (parent.visits as f64).sqrt() / (1f64 + self.visits as f64);
    }
}

pub fn mcts_connect_four(node: Rc<RefCell<ConnectFourState>>, game: &mut ConnectFourGame, model: &ConnectFourModel, display: bool) -> i64 {
    if display {
        game.display();
    }

    node.borrow_mut().visits += 1;

    if node.borrow().moves.is_none() {
        let valid_moves = game.valid_moves();

        let (policy, score) = model.forward(&game);
        let policy = policy.to_device(Device::Cpu);

        let mut moves: Vec<Rc<RefCell<ConnectFourState>>> = Vec::new();

        for valid_move in valid_moves {
            let mut child = ConnectFourState::new(Some(node.clone()), policy.get(valid_move).double_value(&[]));

            child.game_move = Some(valid_move);

            moves.push(Rc::new(RefCell::new(child)));
        }

        node.borrow_mut().moves = Some(moves);
    }

    let result = game.result();

    if result == 0 && node.borrow().moves.as_ref().unwrap().len() > 0 {
        let mut best_move: Option<Rc<RefCell<ConnectFourState>>> = None;
        let mut best_score = 0f64;

        for game_move in node.borrow().moves.as_ref().unwrap() {
            let game_move_access = game_move.borrow();

            let score = game_move_access.get_score();

            if best_move.is_none() || score > best_score {
                best_move = Some(game_move.clone());
                best_score = score;
            }

            if display {
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

        if display {
            println!("Exploring {}", best_move.as_ref().unwrap().borrow().game_move.unwrap());
        }

        game.make_move(best_move.as_ref().unwrap().borrow().game_move.unwrap());

        let result = mcts_connect_four(best_move.unwrap().clone(), game, model, display);

        game.undo_move();

        {
            node.borrow_mut().score_total += result * -game.perspective;
        }

        return result;
    } else {
        if display {
            println!("Ended with result! {}", result);
        }

        {
            node.borrow_mut().score_total += result * -game.perspective;
        }

        return result;
    }
}
