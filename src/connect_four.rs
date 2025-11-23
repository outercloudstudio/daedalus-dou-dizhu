use tch::nn::{Linear, Module, OptimizerConfig, Path};
use tch::{Device, Kind, Tensor, nn};

pub struct ConnectFourGame {
    board_state: Tensor,
    perspective: i64,
    history: Vec<i64>,
}

impl ConnectFourGame {
    pub fn new() -> Self {
        ConnectFourGame {
            board_state: Tensor::zeros(&[6, 7], (Kind::Float, tch::Device::Cpu)),
            perspective: 1,
            history: Vec::new(),
        }
    }

    pub fn move_valid(&self, position: i64) -> bool {
        self.board_state.get(0).get(position).double_value(&[]) == 0.0
    }

    pub fn make_move(&mut self, position: i64) {
        for i in 0..6 {
            let row = 5 - i;

            if self.board_state.get(row).get(position).double_value(&[]) == 0.0 {
                let _ = self.board_state.get(row).get(position).fill_(1.0);

                break;
            }
        }

        self.history.push(position);
        self.board_state *= -1.0;
        self.perspective *= -1;
    }

    pub fn undo_move(&mut self) {
        let last_move = self.history.pop().unwrap();

        for i in 0..6 {
            if self.board_state.get(i).get(last_move).double_value(&[]) != 0.0 {
                let _ = self.board_state.get(i).get(last_move).fill_(0.0);

                break;
            }
        }

        self.board_state *= -1.0;
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
                let looking_for = self.board_state.get(start_y).get(start_x).double_value(&[]);

                if looking_for == 0.0 {
                    continue;
                }

                let mut found = true;

                for i in 0..3 {
                    if self
                        .board_state
                        .get(start_y)
                        .get(start_x + 1 + i)
                        .double_value(&[])
                        != looking_for
                    {
                        found = false;
                        break;
                    }
                }

                if found {
                    return (looking_for as i64) * self.perspective;
                }
            }
        }

        for start_y in 0..3 {
            for start_x in 0..7 {
                let looking_for = self.board_state.get(start_y).get(start_x).double_value(&[]);

                if looking_for == 0.0 {
                    continue;
                }

                let mut found = true;

                for i in 0..3 {
                    if self
                        .board_state
                        .get(start_y + 1 + i)
                        .get(start_x)
                        .double_value(&[])
                        != looking_for
                    {
                        found = false;
                        break;
                    }
                }

                if found {
                    return (looking_for as i64) * self.perspective;
                }
            }
        }

        for start_y in 0..3 {
            for start_x in 0..4 {
                let looking_for = self.board_state.get(start_y).get(start_x).double_value(&[]);

                if looking_for != 0.0 {
                    let mut found = true;

                    for i in 0..3 {
                        if self
                            .board_state
                            .get(start_y + 1 + i)
                            .get(start_x + 1 + i)
                            .double_value(&[])
                            != looking_for
                        {
                            found = false;
                            break;
                        }
                    }

                    if found {
                        return (looking_for as i64) * self.perspective;
                    }
                }

                let looking_for = self
                    .board_state
                    .get(start_y + 3)
                    .get(start_x)
                    .double_value(&[]);

                if looking_for != 0.0 {
                    let mut found = true;

                    for i in 0..3 {
                        if self
                            .board_state
                            .get(start_y + 2 - i)
                            .get(start_x + 1 + i)
                            .double_value(&[])
                            != looking_for
                        {
                            found = false;
                            break;
                        }
                    }

                    if found {
                        return (looking_for as i64) * self.perspective;
                    }
                }
            }
        }

        return 0;
    }

    pub fn display(&self) {
        println!("-------");

        for y in 0..6 {
            let mut line = String::new();

            for x in 0..7 {
                let value = self.board_state.get(y).get(x).double_value(&[]);

                if value == 0.0 {
                    line.push(' ');
                } else if (value as i64) * self.perspective == 1 {
                    line.push('O');
                } else if (value as i64) * self.perspective == -1 {
                    line.push('X');
                }
            }

            println!("{}", line);
        }

        println!("-------");
    }

    pub fn board_state(&self) -> &Tensor {
        &self.board_state
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

    pub fn forward(&self, board_state: &Tensor) -> (Tensor, Tensor) {
        // Flatten and convert to float
        let mut value = board_state.flatten(0, -1).to_kind(Kind::Float);

        // Forward pass through layers with ReLU activations
        value = value.apply(&self.layer1).relu();
        value = value.apply(&self.layer2).relu();
        value = value.apply(&self.layer3).relu();
        value = value.apply(&self.layer4);

        // Split into policy (first 7) and score (last 1)
        let policy = value.narrow(0, 0, 7);
        let score = value.get(7);

        // Apply softmax to policy and tanh to score
        let policy = policy.softmax(0, Kind::Float);
        let score = score.tanh();

        (policy, score)
    }

    pub fn loss(
        &self,
        policy: &Tensor,
        score: &Tensor,
        target_policy: &Tensor,
        target_score: &Tensor,
    ) -> Tensor {
        let policy_loss = -(target_policy * policy.log()).sum(Kind::Float);

        let value_loss = (score - target_score).pow_tensor_scalar(2);

        policy_loss + value_loss
    }
}
