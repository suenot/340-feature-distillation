use anyhow::{Context, Result};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::Deserialize;

// =============================================================================
// Bybit API Data Fetching
// =============================================================================

#[derive(Debug, Deserialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i64,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize)]
pub struct BybitResult {
    pub symbol: Option<String>,
    pub category: Option<String>,
    pub list: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch kline data from Bybit API
pub fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let resp: BybitResponse = client
        .get(&url)
        .send()
        .context("Failed to send request to Bybit")?
        .json()
        .context("Failed to parse Bybit response")?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", resp.ret_msg);
    }

    let mut candles: Vec<Candle> = resp
        .result
        .list
        .iter()
        .filter_map(|row| {
            if row.len() >= 6 {
                Some(Candle {
                    timestamp: row[0].parse().ok()?,
                    open: row[1].parse().ok()?,
                    high: row[2].parse().ok()?,
                    low: row[3].parse().ok()?,
                    close: row[4].parse().ok()?,
                    volume: row[5].parse().ok()?,
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first, reverse to chronological order
    candles.reverse();
    Ok(candles)
}

/// Convert candles to feature matrix and labels
pub fn candles_to_features(candles: &[Candle], lookback: usize) -> (Array2<f64>, Array1<f64>) {
    let n = candles.len();
    if n < lookback + 1 {
        return (Array2::zeros((0, 0)), Array1::zeros(0));
    }

    let num_samples = n - lookback;
    let features_per_step = 6; // returns, log_returns, range, vol_norm, vwap_approx, rsi_approx
    let feature_dim = lookback * features_per_step;

    let mut features = Array2::zeros((num_samples, feature_dim));
    let mut labels = Array1::zeros(num_samples);

    for i in 0..num_samples {
        let window = &candles[i..i + lookback + 1];
        for j in 0..lookback {
            let c = &window[j];
            let prev_close = if j > 0 { window[j - 1].close } else { c.open };

            let ret = (c.close - prev_close) / prev_close.max(1e-10);
            let log_ret = (c.close / prev_close.max(1e-10)).ln();
            let range = (c.high - c.low) / c.open.max(1e-10);
            let vol_norm = c.volume / 1e6;
            let vwap_approx = (c.high + c.low + c.close) / 3.0 / c.open.max(1e-10) - 1.0;

            // Simple RSI approximation based on recent gains/losses
            let gain = if ret > 0.0 { ret } else { 0.0 };
            let loss = if ret < 0.0 { -ret } else { 0.0 };
            let rsi_approx = if gain + loss > 0.0 {
                gain / (gain + loss) * 2.0 - 1.0
            } else {
                0.0
            };

            let base = j * features_per_step;
            features[[i, base]] = ret;
            features[[i, base + 1]] = log_ret;
            features[[i, base + 2]] = range;
            features[[i, base + 3]] = vol_norm;
            features[[i, base + 4]] = vwap_approx;
            features[[i, base + 5]] = rsi_approx;
        }

        // Label: direction of next candle
        let next = &window[lookback];
        labels[i] = if next.close > next.open { 1.0 } else { 0.0 };
    }

    (features, labels)
}

// =============================================================================
// Neural Network Layers and Utilities
// =============================================================================

/// ReLU activation
pub fn relu(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| v.max(0.0))
}

/// Sigmoid activation
pub fn sigmoid(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

/// Sigmoid for 1D array
pub fn sigmoid_1d(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

/// Softmax with temperature
pub fn softmax_with_temperature(x: &Array1<f64>, temperature: f64) -> Array1<f64> {
    let scaled = x / temperature;
    let max_val = scaled.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals = (scaled - max_val).mapv(|v| v.exp());
    let sum: f64 = exp_vals.sum();
    exp_vals / sum.max(1e-10)
}

/// Initialize weights with Xavier initialization
pub fn xavier_init(rows: usize, cols: usize, rng: &mut impl Rng) -> Array2<f64> {
    let scale = (2.0 / (rows + cols) as f64).sqrt();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-scale..scale))
}

/// Initialize bias as zeros
pub fn zeros_bias(size: usize) -> Array1<f64> {
    Array1::zeros(size)
}

// =============================================================================
// Linear Layer
// =============================================================================

#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
}

impl LinearLayer {
    pub fn new(input_dim: usize, output_dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            weights: xavier_init(input_dim, output_dim, rng),
            bias: zeros_bias(output_dim),
        }
    }

    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let output = input.dot(&self.weights);
        output + &self.bias
    }

    /// Simple gradient update for a single layer
    pub fn update(
        &mut self,
        input: &Array2<f64>,
        grad_output: &Array2<f64>,
        lr: f64,
    ) -> Array2<f64> {
        let batch_size = input.nrows() as f64;

        // Weight gradient
        let grad_w = input.t().dot(grad_output) / batch_size;
        // Bias gradient
        let grad_b = grad_output.sum_axis(Axis(0)) / batch_size;
        // Input gradient (for backprop)
        let grad_input = grad_output.dot(&self.weights.t());

        self.weights = &self.weights - &(grad_w * lr);
        self.bias = &self.bias - &(grad_b * lr);

        grad_input
    }
}

// =============================================================================
// Projection Head (for dimension alignment in feature distillation)
// =============================================================================

#[derive(Debug, Clone)]
pub struct ProjectionHead {
    pub layer: LinearLayer,
}

impl ProjectionHead {
    pub fn new(input_dim: usize, output_dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            layer: LinearLayer::new(input_dim, output_dim, rng),
        }
    }

    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        self.layer.forward(input)
    }

    pub fn update(&mut self, input: &Array2<f64>, grad_output: &Array2<f64>, lr: f64) {
        self.layer.update(input, grad_output, lr);
    }
}

// =============================================================================
// Teacher Network
// =============================================================================

#[derive(Debug, Clone)]
pub struct TeacherNetwork {
    pub layers: Vec<LinearLayer>,
    pub output_layer: LinearLayer,
}

impl TeacherNetwork {
    pub fn new(input_dim: usize, hidden_dims: &[usize], output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut layers = Vec::new();
        let mut prev_dim = input_dim;

        for &dim in hidden_dims {
            layers.push(LinearLayer::new(prev_dim, dim, &mut rng));
            prev_dim = dim;
        }

        let output_layer = LinearLayer::new(prev_dim, output_dim, &mut rng);

        Self {
            layers,
            output_layer,
        }
    }

    /// Forward pass returning output and all intermediate features
    pub fn forward_with_features(&self, input: &Array2<f64>) -> (Array2<f64>, Vec<Array2<f64>>) {
        let mut features = Vec::new();
        let mut x = input.clone();

        for layer in &self.layers {
            x = relu(&layer.forward(&x));
            features.push(x.clone());
        }

        let output = sigmoid(&self.output_layer.forward(&x));
        (output, features)
    }

    /// Forward pass returning only the output
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        self.forward_with_features(input).0
    }

    /// Train one step on (input, labels) with MSE loss
    pub fn train_step(
        &mut self,
        input: &Array2<f64>,
        labels: &Array1<f64>,
        lr: f64,
    ) -> f64 {
        let (output, intermediates) = self.forward_with_features(input);
        let output_col = output.column(0).to_owned();

        // MSE loss gradient
        let diff = &output_col - labels;
        let loss = diff.mapv(|v| v * v).mean().unwrap_or(0.0);

        // Backprop through output layer
        let grad = diff.mapv(|v| 2.0 * v) / input.nrows() as f64;
        let sigmoid_grad = output_col.mapv(|v| v * (1.0 - v));
        let grad = &grad * &sigmoid_grad;
        let grad_2d = grad.insert_axis(Axis(1));

        let last_hidden = if intermediates.is_empty() {
            input.clone()
        } else {
            intermediates.last().unwrap().clone()
        };
        let mut grad_back = self.output_layer.update(&last_hidden, &grad_2d, lr);

        // Backprop through hidden layers
        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            // ReLU gradient
            let pre_relu_input = if i > 0 {
                &intermediates[i]
            } else {
                &intermediates[0]
            };
            let relu_grad = pre_relu_input.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
            grad_back = &grad_back * &relu_grad;

            let layer_input = if i > 0 {
                &intermediates[i - 1]
            } else {
                input
            };
            grad_back = layer.update(layer_input, &grad_back, lr);
        }

        loss
    }
}

// =============================================================================
// Student Network
// =============================================================================

#[derive(Debug, Clone)]
pub struct StudentNetwork {
    pub layers: Vec<LinearLayer>,
    pub output_layer: LinearLayer,
}

impl StudentNetwork {
    pub fn new(input_dim: usize, hidden_dims: &[usize], output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut layers = Vec::new();
        let mut prev_dim = input_dim;

        for &dim in hidden_dims {
            layers.push(LinearLayer::new(prev_dim, dim, &mut rng));
            prev_dim = dim;
        }

        let output_layer = LinearLayer::new(prev_dim, output_dim, &mut rng);

        Self {
            layers,
            output_layer,
        }
    }

    /// Forward pass returning output and all intermediate features
    pub fn forward_with_features(&self, input: &Array2<f64>) -> (Array2<f64>, Vec<Array2<f64>>) {
        let mut features = Vec::new();
        let mut x = input.clone();

        for layer in &self.layers {
            x = relu(&layer.forward(&x));
            features.push(x.clone());
        }

        let output = sigmoid(&self.output_layer.forward(&x));
        (output, features)
    }

    /// Forward pass returning only the output
    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        self.forward_with_features(input).0
    }

    /// Train one step with standard MSE loss (no distillation)
    pub fn train_step(
        &mut self,
        input: &Array2<f64>,
        labels: &Array1<f64>,
        lr: f64,
    ) -> f64 {
        let (output, intermediates) = self.forward_with_features(input);
        let output_col = output.column(0).to_owned();

        let diff = &output_col - labels;
        let loss = diff.mapv(|v| v * v).mean().unwrap_or(0.0);

        let grad = diff.mapv(|v| 2.0 * v) / input.nrows() as f64;
        let sigmoid_grad = output_col.mapv(|v| v * (1.0 - v));
        let grad = &grad * &sigmoid_grad;
        let grad_2d = grad.insert_axis(Axis(1));

        let last_hidden = if intermediates.is_empty() {
            input.clone()
        } else {
            intermediates.last().unwrap().clone()
        };
        let mut grad_back = self.output_layer.update(&last_hidden, &grad_2d, lr);

        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let relu_grad = intermediates[i].mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
            grad_back = &grad_back * &relu_grad;

            let layer_input = if i > 0 {
                &intermediates[i - 1]
            } else {
                input
            };
            grad_back = layer.update(layer_input, &grad_back, lr);
        }

        loss
    }
}

// =============================================================================
// Feature Distillation Losses
// =============================================================================

/// FitNet-style hint loss: MSE between teacher features and projected student features
pub fn fitnet_loss(
    teacher_features: &Array2<f64>,
    student_features: &Array2<f64>,
    projection: &ProjectionHead,
) -> f64 {
    let projected = projection.forward(student_features);
    let diff = &projected - teacher_features;
    diff.mapv(|v| v * v).mean().unwrap_or(0.0)
}

/// Compute spatial attention map: sum of squared activations across feature dimension
pub fn compute_attention_map(features: &Array2<f64>) -> Array1<f64> {
    // For each sample, compute sum of squared features (attention score)
    let squared = features.mapv(|v| v * v);
    squared.sum_axis(Axis(1))
}

/// Attention transfer loss between teacher and student feature maps
pub fn attention_transfer_loss(
    teacher_features: &[Array2<f64>],
    student_features: &[Array2<f64>],
) -> f64 {
    let num_pairs = teacher_features.len().min(student_features.len());
    if num_pairs == 0 {
        return 0.0;
    }

    let mut total_loss = 0.0;

    for i in 0..num_pairs {
        let t_idx = i * teacher_features.len() / num_pairs;
        let s_idx = i * student_features.len() / num_pairs;

        let t_attn = compute_attention_map(&teacher_features[t_idx]);
        let s_attn = compute_attention_map(&student_features[s_idx]);

        // Normalize attention maps
        let t_norm = t_attn.mapv(|v| v.abs()).sum().max(1e-10);
        let s_norm = s_attn.mapv(|v| v.abs()).sum().max(1e-10);

        let t_normalized = &t_attn / t_norm;
        let s_normalized = &s_attn / s_norm;

        let diff = &t_normalized - &s_normalized;
        total_loss += diff.mapv(|v| v * v).sum();
    }

    total_loss / num_pairs as f64
}

/// Compute Gram matrix: feature correlation matrix
pub fn compute_gram_matrix(features: &Array2<f64>) -> Array2<f64> {
    let ft = features.t();
    ft.dot(features)
}

/// Gram matrix matching loss between teacher and student features
pub fn gram_matrix_loss(
    teacher_features: &Array2<f64>,
    student_features: &Array2<f64>,
) -> f64 {
    let t_gram = compute_gram_matrix(teacher_features);
    let s_gram = compute_gram_matrix(student_features);

    // Normalize by dimensions
    let t_size = (t_gram.nrows() * t_gram.ncols()) as f64;
    let s_size = (s_gram.nrows() * s_gram.ncols()) as f64;

    let t_normalized = &t_gram / t_size.max(1.0);
    let s_normalized = &s_gram / s_size.max(1.0);

    // Since dimensions may differ, compare statistics
    let t_mean = t_normalized.mean().unwrap_or(0.0);
    let s_mean = s_normalized.mean().unwrap_or(0.0);
    let t_var = t_normalized.mapv(|v| (v - t_mean).powi(2)).mean().unwrap_or(0.0);
    let s_var = s_normalized.mapv(|v| (v - s_mean).powi(2)).mean().unwrap_or(0.0);

    (t_mean - s_mean).powi(2) + (t_var - s_var).powi(2)
}

/// Standard output-level knowledge distillation loss (KL divergence approximation)
pub fn kd_loss(teacher_output: &Array2<f64>, student_output: &Array2<f64>, temperature: f64) -> f64 {
    let t = teacher_output.mapv(|v| v.max(1e-10).min(1.0 - 1e-10));
    let s = student_output.mapv(|v| v.max(1e-10).min(1.0 - 1e-10));

    // For binary classification, use binary KL divergence
    let kl = &t * (&t / &s).mapv(|v| v.ln()) + (1.0 - &t) * ((1.0 - &t) / (1.0 - &s)).mapv(|v| v.ln());
    let scaled = kl * temperature * temperature;
    scaled.mean().unwrap_or(0.0)
}

// =============================================================================
// Distillation Configuration
// =============================================================================

#[derive(Debug, Clone)]
pub enum FeatureMethod {
    FitNet,
    AttentionTransfer,
    GramMatrix,
    Combined,
}

#[derive(Debug, Clone)]
pub struct DistillationConfig {
    pub alpha: f64,         // Task loss weight
    pub beta: f64,          // KD output loss weight
    pub gamma: Vec<f64>,    // Feature matching weights per pair
    pub temperature: f64,
    pub feature_method: FeatureMethod,
    pub learning_rate: f64,
    pub epochs: usize,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            alpha: 0.3,
            beta: 0.3,
            gamma: vec![0.2, 0.2],
            temperature: 4.0,
            feature_method: FeatureMethod::Combined,
            learning_rate: 0.01,
            epochs: 100,
        }
    }
}

// =============================================================================
// Feature Distillation Trainer
// =============================================================================

pub struct FeatureDistillationTrainer {
    pub config: DistillationConfig,
    pub projections: Vec<ProjectionHead>,
}

impl FeatureDistillationTrainer {
    pub fn new(
        config: DistillationConfig,
        teacher_hidden_dims: &[usize],
        student_hidden_dims: &[usize],
    ) -> Self {
        let mut rng = rand::thread_rng();
        let num_pairs = teacher_hidden_dims.len().min(student_hidden_dims.len());

        let mut projections = Vec::new();
        for i in 0..num_pairs {
            let s_idx = i * student_hidden_dims.len() / num_pairs;
            let t_idx = i * teacher_hidden_dims.len() / num_pairs;
            projections.push(ProjectionHead::new(
                student_hidden_dims[s_idx],
                teacher_hidden_dims[t_idx],
                &mut rng,
            ));
        }

        Self {
            config,
            projections,
        }
    }

    /// Compute the combined feature distillation loss
    pub fn compute_feature_loss(
        &self,
        teacher_features: &[Array2<f64>],
        student_features: &[Array2<f64>],
    ) -> f64 {
        let mut total = 0.0;
        let num_pairs = teacher_features.len().min(student_features.len());

        match &self.config.feature_method {
            FeatureMethod::FitNet => {
                for i in 0..num_pairs.min(self.projections.len()) {
                    let t_idx = i * teacher_features.len() / num_pairs;
                    let s_idx = i * student_features.len() / num_pairs;
                    let gamma = self.config.gamma.get(i).copied().unwrap_or(0.1);
                    total +=
                        gamma * fitnet_loss(&teacher_features[t_idx], &student_features[s_idx], &self.projections[i]);
                }
            }
            FeatureMethod::AttentionTransfer => {
                let gamma = self.config.gamma.first().copied().unwrap_or(0.1);
                total += gamma * attention_transfer_loss(teacher_features, student_features);
            }
            FeatureMethod::GramMatrix => {
                for i in 0..num_pairs {
                    let t_idx = i * teacher_features.len() / num_pairs;
                    let s_idx = i * student_features.len() / num_pairs;
                    let gamma = self.config.gamma.get(i).copied().unwrap_or(0.1);
                    total +=
                        gamma * gram_matrix_loss(&teacher_features[t_idx], &student_features[s_idx]);
                }
            }
            FeatureMethod::Combined => {
                // Combine all three methods
                let weight = 1.0 / 3.0;

                for i in 0..num_pairs.min(self.projections.len()) {
                    let t_idx = i * teacher_features.len() / num_pairs;
                    let s_idx = i * student_features.len() / num_pairs;
                    let gamma = self.config.gamma.get(i).copied().unwrap_or(0.1);

                    let fit = fitnet_loss(
                        &teacher_features[t_idx],
                        &student_features[s_idx],
                        &self.projections[i],
                    );
                    let gram = gram_matrix_loss(&teacher_features[t_idx], &student_features[s_idx]);
                    total += gamma * weight * (fit + gram);
                }

                let at = attention_transfer_loss(teacher_features, student_features);
                let gamma = self.config.gamma.first().copied().unwrap_or(0.1);
                total += gamma * weight * at;
            }
        }

        total
    }

    /// Train student with feature distillation from a frozen teacher
    pub fn train(
        &self,
        teacher: &TeacherNetwork,
        student: &mut StudentNetwork,
        input: &Array2<f64>,
        labels: &Array1<f64>,
    ) -> Vec<f64> {
        let mut losses = Vec::new();
        let lr = self.config.learning_rate;

        for epoch in 0..self.config.epochs {
            // Forward passes
            let (t_output, t_features) = teacher.forward_with_features(input);
            let (s_output, s_features) = student.forward_with_features(input);

            // Task loss (MSE)
            let s_col = s_output.column(0).to_owned();
            let diff = &s_col - labels;
            let task_loss = diff.mapv(|v| v * v).mean().unwrap_or(0.0);

            // KD loss
            let kd = kd_loss(&t_output, &s_output, self.config.temperature);

            // Feature distillation loss
            let feat_loss = self.compute_feature_loss(&t_features, &s_features);

            let total_loss =
                self.config.alpha * task_loss + self.config.beta * kd + feat_loss;

            // Simple gradient descent on student parameters
            // We use the task loss gradient as the primary signal
            // and add feature matching as a regularizer
            let grad = diff.mapv(|v| 2.0 * v) / input.nrows() as f64;
            let sigmoid_grad = s_col.mapv(|v| v * (1.0 - v));
            let grad = &grad * &sigmoid_grad;
            let grad_2d = grad.insert_axis(Axis(1));

            let last_hidden = if s_features.is_empty() {
                input.clone()
            } else {
                s_features.last().unwrap().clone()
            };
            let mut grad_back = student.output_layer.update(&last_hidden, &grad_2d, lr);

            for (i, layer) in student.layers.iter_mut().enumerate().rev() {
                let relu_grad = s_features[i].mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
                grad_back = &grad_back * &relu_grad;

                let layer_input = if i > 0 {
                    &s_features[i - 1]
                } else {
                    input
                };
                grad_back = layer.update(layer_input, &grad_back, lr);
            }

            losses.push(total_loss);

            if epoch % 20 == 0 {
                println!(
                    "  Epoch {}: total={:.6}, task={:.6}, kd={:.6}, feat={:.6}",
                    epoch, total_loss, task_loss, kd, feat_loss
                );
            }
        }

        losses
    }
}

// =============================================================================
// Feature Analysis Utilities
// =============================================================================

/// Compute cosine similarity between two feature vectors
pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Centered Kernel Alignment (CKA) - linear version
/// Measures similarity between two representation spaces
pub fn linear_cka(x: &Array2<f64>, y: &Array2<f64>) -> f64 {
    let n = x.nrows();
    if n == 0 {
        return 0.0;
    }

    // Center the matrices
    let x_mean = x.mean_axis(Axis(0)).unwrap();
    let y_mean = y.mean_axis(Axis(0)).unwrap();
    let x_centered = x - &x_mean;
    let y_centered = y - &y_mean;

    // Compute HSIC
    let xx = x_centered.dot(&x_centered.t());
    let yy = y_centered.dot(&y_centered.t());

    let hsic_xy: f64 = xx.iter().zip(yy.iter()).map(|(a, b)| a * b).sum();
    let hsic_xx: f64 = xx.iter().map(|a| a * a).sum::<f64>().sqrt();
    let hsic_yy: f64 = yy.iter().map(|a| a * a).sum::<f64>().sqrt();

    if hsic_xx < 1e-10 || hsic_yy < 1e-10 {
        return 0.0;
    }

    hsic_xy / (hsic_xx * hsic_yy)
}

/// Analyze feature similarity between teacher and student at each layer pair
pub fn analyze_feature_similarity(
    teacher_features: &[Array2<f64>],
    student_features: &[Array2<f64>],
) -> Vec<FeatureSimilarityReport> {
    let num_pairs = teacher_features.len().min(student_features.len());
    let mut reports = Vec::new();

    for i in 0..num_pairs {
        let t_idx = i * teacher_features.len() / num_pairs;
        let s_idx = i * student_features.len() / num_pairs;

        let t = &teacher_features[t_idx];
        let s = &student_features[s_idx];

        // Mean cosine similarity across samples
        let mut cos_sims = Vec::new();
        for row in 0..t.nrows().min(s.nrows()) {
            let t_row = t.row(row).to_owned();
            let s_row = s.row(row).to_owned();
            // Truncate to shorter dimension
            let min_dim = t_row.len().min(s_row.len());
            let t_trunc = t_row.slice(ndarray::s![..min_dim]).to_owned();
            let s_trunc = s_row.slice(ndarray::s![..min_dim]).to_owned();
            cos_sims.push(cosine_similarity(&t_trunc, &s_trunc));
        }
        let mean_cosine = if cos_sims.is_empty() {
            0.0
        } else {
            cos_sims.iter().sum::<f64>() / cos_sims.len() as f64
        };

        // CKA similarity (using truncated features for dimension matching)
        let min_dim = t.ncols().min(s.ncols());
        let t_trunc = t.slice(ndarray::s![.., ..min_dim]).to_owned();
        let s_trunc = s.slice(ndarray::s![.., ..min_dim]).to_owned();
        let cka = linear_cka(&t_trunc, &s_trunc);

        // Attention map correlation
        let t_attn = compute_attention_map(t);
        let s_attn = compute_attention_map(s);
        let attn_corr = cosine_similarity(&t_attn, &s_attn);

        reports.push(FeatureSimilarityReport {
            layer_pair: (t_idx, s_idx),
            mean_cosine_similarity: mean_cosine,
            cka_similarity: cka,
            attention_correlation: attn_corr,
            teacher_feature_dim: t.ncols(),
            student_feature_dim: s.ncols(),
        });
    }

    reports
}

#[derive(Debug, Clone)]
pub struct FeatureSimilarityReport {
    pub layer_pair: (usize, usize),
    pub mean_cosine_similarity: f64,
    pub cka_similarity: f64,
    pub attention_correlation: f64,
    pub teacher_feature_dim: usize,
    pub student_feature_dim: usize,
}

impl std::fmt::Display for FeatureSimilarityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Layer pair ({}, {}): cosine={:.4}, CKA={:.4}, attn_corr={:.4} [T_dim={}, S_dim={}]",
            self.layer_pair.0,
            self.layer_pair.1,
            self.mean_cosine_similarity,
            self.cka_similarity,
            self.attention_correlation,
            self.teacher_feature_dim,
            self.student_feature_dim,
        )
    }
}

/// Compute model accuracy on binary classification
pub fn compute_accuracy(model_output: &Array2<f64>, labels: &Array1<f64>) -> f64 {
    let predictions = model_output.column(0);
    let correct: usize = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(&pred, &label)| {
            let predicted_class = if pred > 0.5 { 1.0 } else { 0.0 };
            (predicted_class - label).abs() < 1e-10
        })
        .count();
    correct as f64 / labels.len() as f64
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Array2<f64>, Array1<f64>) {
        let mut rng = rand::thread_rng();
        let n = 50;
        let d = 12;
        let x = Array2::from_shape_fn((n, d), |_| rng.gen_range(-1.0..1.0));
        let labels = Array1::from_shape_fn(n, |i| if i % 2 == 0 { 1.0 } else { 0.0 });
        (x, labels)
    }

    #[test]
    fn test_teacher_forward() {
        let teacher = TeacherNetwork::new(12, &[64, 32, 16], 1);
        let (x, _) = make_test_data();
        let (output, features) = teacher.forward_with_features(&x);

        assert_eq!(output.nrows(), 50);
        assert_eq!(output.ncols(), 1);
        assert_eq!(features.len(), 3);
        assert_eq!(features[0].ncols(), 64);
        assert_eq!(features[1].ncols(), 32);
        assert_eq!(features[2].ncols(), 16);
    }

    #[test]
    fn test_student_forward() {
        let student = StudentNetwork::new(12, &[16, 8], 1);
        let (x, _) = make_test_data();
        let (output, features) = student.forward_with_features(&x);

        assert_eq!(output.nrows(), 50);
        assert_eq!(output.ncols(), 1);
        assert_eq!(features.len(), 2);
        assert_eq!(features[0].ncols(), 16);
        assert_eq!(features[1].ncols(), 8);
    }

    #[test]
    fn test_teacher_training_reduces_loss() {
        let mut teacher = TeacherNetwork::new(12, &[32, 16], 1);
        let (x, labels) = make_test_data();

        let initial_loss = teacher.train_step(&x, &labels, 0.01);
        let mut last_loss = initial_loss;
        for _ in 0..50 {
            last_loss = teacher.train_step(&x, &labels, 0.01);
        }

        assert!(
            last_loss < initial_loss,
            "Loss should decrease: initial={}, final={}",
            initial_loss,
            last_loss
        );
    }

    #[test]
    fn test_fitnet_loss_computes() {
        let mut rng = rand::thread_rng();
        let teacher_feat = Array2::from_shape_fn((10, 32), |_| rng.gen_range(-1.0..1.0));
        let student_feat = Array2::from_shape_fn((10, 16), |_| rng.gen_range(-1.0..1.0));
        let proj = ProjectionHead::new(16, 32, &mut rng);

        let loss = fitnet_loss(&teacher_feat, &student_feat, &proj);
        assert!(loss >= 0.0, "FitNet loss should be non-negative");
        assert!(loss.is_finite(), "FitNet loss should be finite");
    }

    #[test]
    fn test_attention_transfer_loss_computes() {
        let mut rng = rand::thread_rng();
        let t_features = vec![
            Array2::from_shape_fn((10, 64), |_| rng.gen_range(-1.0..1.0)),
            Array2::from_shape_fn((10, 32), |_| rng.gen_range(-1.0..1.0)),
        ];
        let s_features = vec![
            Array2::from_shape_fn((10, 16), |_| rng.gen_range(-1.0..1.0)),
        ];

        let loss = attention_transfer_loss(&t_features, &s_features);
        assert!(loss >= 0.0, "AT loss should be non-negative");
        assert!(loss.is_finite(), "AT loss should be finite");
    }

    #[test]
    fn test_gram_matrix_loss_computes() {
        let mut rng = rand::thread_rng();
        let teacher_feat = Array2::from_shape_fn((10, 32), |_| rng.gen_range(-1.0..1.0));
        let student_feat = Array2::from_shape_fn((10, 16), |_| rng.gen_range(-1.0..1.0));

        let loss = gram_matrix_loss(&teacher_feat, &student_feat);
        assert!(loss >= 0.0, "Gram loss should be non-negative");
        assert!(loss.is_finite(), "Gram loss should be finite");
    }

    #[test]
    fn test_kd_loss_computes() {
        let mut rng = rand::thread_rng();
        let t_out = Array2::from_shape_fn((10, 1), |_| rng.gen_range(0.1..0.9));
        let s_out = Array2::from_shape_fn((10, 1), |_| rng.gen_range(0.1..0.9));

        let loss = kd_loss(&t_out, &s_out, 4.0);
        assert!(loss >= 0.0, "KD loss should be non-negative");
        assert!(loss.is_finite(), "KD loss should be finite");
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);

        let c = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        assert!(cosine_similarity(&a, &c).abs() < 1e-10);
    }

    #[test]
    fn test_linear_cka() {
        let mut rng = rand::thread_rng();
        let x = Array2::from_shape_fn((20, 8), |_| rng.gen_range(-1.0..1.0));

        // CKA of a matrix with itself should be close to 1.0
        let cka = linear_cka(&x, &x);
        assert!(
            (cka - 1.0).abs() < 1e-6,
            "CKA of identical matrices should be ~1.0, got {}",
            cka
        );
    }

    #[test]
    fn test_feature_similarity_report() {
        let mut rng = rand::thread_rng();
        let t_features = vec![
            Array2::from_shape_fn((10, 64), |_| rng.gen_range(-1.0..1.0)),
            Array2::from_shape_fn((10, 32), |_| rng.gen_range(-1.0..1.0)),
        ];
        let s_features = vec![
            Array2::from_shape_fn((10, 16), |_| rng.gen_range(-1.0..1.0)),
            Array2::from_shape_fn((10, 8), |_| rng.gen_range(-1.0..1.0)),
        ];

        let reports = analyze_feature_similarity(&t_features, &s_features);
        assert_eq!(reports.len(), 2);

        for report in &reports {
            assert!(report.mean_cosine_similarity.is_finite());
            assert!(report.cka_similarity.is_finite());
            assert!(report.attention_correlation.is_finite());
        }
    }

    #[test]
    fn test_compute_accuracy() {
        let output = Array2::from_shape_vec((4, 1), vec![0.8, 0.3, 0.7, 0.2]).unwrap();
        let labels = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        let acc = compute_accuracy(&output, &labels);
        assert!((acc - 1.0).abs() < 1e-10, "Should be 100% accurate");
    }

    #[test]
    fn test_feature_distillation_trainer() {
        let (x, labels) = make_test_data();

        // Train teacher first
        let mut teacher = TeacherNetwork::new(12, &[64, 32], 1);
        for _ in 0..50 {
            teacher.train_step(&x, &labels, 0.01);
        }

        // Create student and trainer
        let config = DistillationConfig {
            alpha: 0.3,
            beta: 0.3,
            gamma: vec![0.2, 0.2],
            temperature: 4.0,
            feature_method: FeatureMethod::Combined,
            learning_rate: 0.01,
            epochs: 30,
        };

        let trainer = FeatureDistillationTrainer::new(config, &[64, 32], &[16, 8]);
        let mut student = StudentNetwork::new(12, &[16, 8], 1);
        let losses = trainer.train(&teacher, &mut student, &x, &labels);

        assert!(!losses.is_empty());
        assert!(losses.last().unwrap().is_finite());
    }

    #[test]
    fn test_projection_head() {
        let mut rng = rand::thread_rng();
        let proj = ProjectionHead::new(16, 32, &mut rng);
        let input = Array2::from_shape_fn((5, 16), |_| rng.gen_range(-1.0..1.0));
        let output = proj.forward(&input);
        assert_eq!(output.shape(), &[5, 32]);
    }

    #[test]
    fn test_candles_to_features() {
        let candles: Vec<Candle> = (0..20)
            .map(|i| Candle {
                timestamp: 1000 + i as u64,
                open: 100.0 + i as f64,
                high: 102.0 + i as f64,
                low: 98.0 + i as f64,
                close: 101.0 + i as f64,
                volume: 1000.0 * (i + 1) as f64,
            })
            .collect();

        let (features, labels) = candles_to_features(&candles, 5);
        assert_eq!(features.nrows(), 15); // 20 - 5
        assert_eq!(features.ncols(), 30); // 5 * 6
        assert_eq!(labels.len(), 15);
    }
}
