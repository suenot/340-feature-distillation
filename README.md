# Chapter 210: Feature Distillation

## 1. Introduction

Knowledge distillation, in its most basic form, transfers the final output distribution from a large teacher model to a smaller student model. While effective, this approach only captures the teacher's final decision-making behavior and discards the rich intermediate representations that the teacher has learned. Feature distillation goes deeper: it transfers the teacher's intermediate representations — the internal features, attention patterns, and activation structures — to the student, enabling the student to learn not just *what* to predict but *how* to reason about the input data.

In the context of trading, this distinction is critical. A large teacher model trained on high-frequency market microstructure data may learn hierarchical features: low-level patterns such as bid-ask spread dynamics, mid-level features like order flow imbalances, and high-level representations capturing regime transitions. Standard output distillation would only transfer the final trading signal. Feature distillation, by contrast, can transfer each of these representational layers, giving the student a much richer understanding of market dynamics despite its smaller size.

Feature distillation encompasses several techniques developed over the past decade: FitNets introduced hint-based training where student intermediate layers are guided to mimic teacher intermediate layers; attention transfer methods match spatial attention maps between teacher and student; Gram matrix matching captures second-order feature statistics; and Contrastive Representation Distillation (CRD) uses contrastive learning to align feature spaces. Each technique offers different trade-offs in computational cost, representational fidelity, and generalization performance.

This chapter provides a comprehensive treatment of feature distillation methods, their mathematical foundations, and their application to algorithmic trading. We implement a full feature distillation framework in Rust with integration to the Bybit exchange API, demonstrating how these techniques can produce compact, deployable trading models that retain the representational power of much larger architectures.

## 2. Mathematical Foundation

### 2.1 FitNets and Hint Layers

FitNets, introduced by Romero et al. (2015), extend knowledge distillation by adding a hint-based training stage. The core idea is to select a "hint layer" from the teacher and a "guided layer" from the student, then train the student so that its guided layer output matches the teacher's hint layer output.

Let $T_h(x)$ denote the output of the teacher's hint layer for input $x$, and $S_g(x)$ denote the output of the student's guided layer. Since these layers may have different dimensions, a learnable projection (regressor) $r$ is introduced:

$$\mathcal{L}_{\text{hint}} = \frac{1}{2} \| T_h(x) - r(S_g(x)) \|^2_2$$

where $r: \mathbb{R}^{d_s} \rightarrow \mathbb{R}^{d_t}$ is a linear or shallow nonlinear mapping that aligns the student's feature space to the teacher's feature space.

Training proceeds in two stages:
1. **Hint training**: minimize $\mathcal{L}_{\text{hint}}$ to pre-train the student's lower layers
2. **Full training**: minimize the standard knowledge distillation loss using soft targets

### 2.2 Attention Transfer

Attention transfer, proposed by Zagoruyko and Komodakis (2017), matches attention maps rather than raw activations. Given a 3D activation tensor $A \in \mathbb{R}^{C \times H \times W}$ from a network layer, the spatial attention map is defined as:

$$Q(A) = \sum_{i=1}^{C} |A_i|^p$$

where $A_i$ is the $i$-th channel and $p$ is a power parameter (typically $p = 2$). The attention transfer loss is:

$$\mathcal{L}_{\text{AT}} = \sum_{j \in \mathcal{I}} \left\| \frac{Q_S^j}{\| Q_S^j \|_2} - \frac{Q_T^j}{\| Q_T^j \|_2} \right\|_2$$

where $\mathcal{I}$ indexes the matched layer pairs, and $Q_S^j$, $Q_T^j$ are attention maps from the student and teacher at layer pair $j$.

For time-series trading data, attention maps capture which temporal positions the model focuses on. Transferring these maps teaches the student to attend to the same market events (e.g., volume spikes, price reversals) as the teacher.

### 2.3 Activation Matching

Direct activation matching aligns the raw activations between teacher and student layers. For matched layer pair $(l_T, l_S)$:

$$\mathcal{L}_{\text{act}} = \frac{1}{N} \sum_{i=1}^{N} \| \phi(F_T^{l_T}(x_i)) - \psi(F_S^{l_S}(x_i)) \|^2_F$$

where $F_T^{l_T}$ and $F_S^{l_S}$ are the activations at the specified layers, $\phi$ and $\psi$ are optional projection functions, and $\| \cdot \|_F$ is the Frobenius norm.

### 2.4 Gram Matrix Matching

Gram matrix matching captures second-order statistics of feature representations. For activation $A \in \mathbb{R}^{C \times D}$ (where $D$ is the spatial or temporal dimension), the Gram matrix is:

$$G(A) = A \cdot A^T \in \mathbb{R}^{C \times C}$$

Each entry $G_{ij}$ measures the correlation between channels $i$ and $j$. The Gram matrix matching loss is:

$$\mathcal{L}_{\text{Gram}} = \sum_{j \in \mathcal{I}} \left\| \frac{G(A_T^j)}{C_T^j \cdot D_T^j} - \frac{G(A_S^j)}{C_S^j \cdot D_S^j} \right\|^2_F$$

This loss captures the "style" of feature activations — the correlational structure between channels — rather than their exact spatial arrangement. In trading, this can capture the structural relationships between learned feature detectors (e.g., the correlation between a trend detector and a volatility detector).

### 2.5 Contrastive Representation Distillation (CRD)

CRD, proposed by Tian et al. (2020), formulates feature distillation as a contrastive learning problem. Instead of directly matching features, CRD maximizes the mutual information between teacher and student representations.

Given teacher representation $t = f_T(x)$ and student representation $s = f_S(x)$, CRD optimizes:

$$\mathcal{L}_{\text{CRD}} = -\mathbb{E}\left[\log \frac{h(t, s)}{\sum_{s' \in \mathcal{N}} h(t, s')}\right]$$

where $h(t, s) = \exp(t^T W s / \tau)$ is a bilinear similarity function, $\mathcal{N}$ is a set of negative samples, and $\tau$ is a temperature parameter. This approach is more flexible than direct matching because it preserves the structural relationships in the representation space without requiring exact alignment.

## 3. Feature Matching Strategies

### 3.1 Which Layers to Match

The choice of which teacher and student layers to match significantly impacts distillation quality. Several strategies exist:

- **Output-of-block matching**: Match the output of each residual block or major architectural unit. This is the most common approach and works well when teacher and student share similar architectures.
- **Semantically meaningful layers**: Choose layers that correspond to interpretable feature levels (e.g., low-level price patterns, mid-level trend features, high-level regime representations).
- **Progressive matching**: Match deeper layers first, then add shallower layers progressively. This prevents the student from being overwhelmed by too many constraints early in training.
- **Attention-guided selection**: Use the teacher's attention patterns to identify which layers carry the most informative features, then prioritize matching those layers.

For trading models, a practical heuristic is to match at three levels: (1) after the initial feature extraction layer (raw market feature processing), (2) at the midpoint of the network (composite feature representations), and (3) just before the final prediction head (high-level market state representation).

### 3.2 Projection Heads for Dimension Alignment

When teacher and student layers have different dimensions, projection heads bridge the gap. Several designs are common:

- **Linear projection**: $r(x) = Wx + b$. Simple and effective for small dimension mismatches.
- **MLP projection**: A two-layer network $r(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$. Better for large dimension mismatches.
- **1x1 convolution**: For feature maps, a 1x1 convolution adjusts the channel dimension while preserving spatial/temporal structure.
- **Adaptive pooling + linear**: First pool the spatial/temporal dimension to match, then apply a linear transformation for the channel dimension.

The projection head is trained jointly with the student but discarded at inference time — it is purely a training artifact.

### 3.3 Loss Weighting

The total training loss combines multiple objectives:

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{task}} + \beta \cdot \mathcal{L}_{\text{KD}} + \sum_{j} \gamma_j \cdot \mathcal{L}_{\text{feature}}^j$$

where $\mathcal{L}_{\text{task}}$ is the standard task loss (e.g., MSE for regression, cross-entropy for classification), $\mathcal{L}_{\text{KD}}$ is the output-level distillation loss, and $\mathcal{L}_{\text{feature}}^j$ is the feature matching loss at layer pair $j$.

Key considerations for loss weighting:

- Feature matching losses at different layers may have different scales; normalize each by the feature dimension to keep them comparable.
- Start with larger feature matching weights and anneal them during training, allowing the student to first learn the teacher's representations and then fine-tune for the task.
- In trading, task loss should never be dominated by feature matching losses — the final prediction quality matters most.

## 4. Trading Applications

### 4.1 Distilling Feature Extractors for Market Microstructure

Market microstructure models often operate on high-dimensional order book data with complex feature hierarchies. A large teacher model might process the full Level 3 order book with hundreds of features, learning representations that capture:

- **Level 1**: Raw order flow statistics (bid-ask spread, queue lengths, trade arrival rates)
- **Level 2**: Derived microstructure signals (order flow imbalance, trade informativeness, adverse selection risk)
- **Level 3**: Regime-aware composite features (liquidity regime, information regime, volatility regime)

Feature distillation can transfer these hierarchical representations to a smaller student that operates on a reduced feature set. The student learns to reconstruct the teacher's internal representations from fewer inputs, effectively learning an efficient encoding of market microstructure.

### 4.2 Transferring Representations Across Timeframes

Features learned at one timeframe often contain information relevant to other timeframes. A teacher trained on tick-level data learns fine-grained representations that can be distilled into a student operating on minute-level data. The feature distillation process teaches the student to extract from minute bars the same representational content that the teacher extracts from ticks.

This is particularly valuable because:
- Tick-level models are expensive to run in production
- Minute-level models are cheaper but miss microstructure information
- Feature distillation bridges this gap by teaching minute-level models to implicitly represent microstructure dynamics

### 4.3 Transferring Representations Across Assets

A teacher trained on liquid assets (e.g., BTCUSDT) develops robust feature representations that may transfer to less liquid assets with limited training data. Feature distillation can transfer these representations, providing the student with a strong inductive bias for understanding market dynamics even when trained on a less liquid market.

## 5. Cross-Domain Feature Transfer

Cross-domain feature transfer applies feature distillation across different markets or asset classes. The key insight is that certain market dynamics are universal: mean reversion, momentum, volatility clustering, and liquidity effects appear across equities, futures, forex, and crypto markets. A teacher model that has learned to detect these dynamics in one market can transfer its feature detectors to a student operating in another market.

The process involves:

1. **Training a teacher** on a data-rich source domain (e.g., US equities with decades of data)
2. **Selecting transferable features**: Use domain adaptation techniques to identify which teacher features are domain-invariant vs. domain-specific
3. **Feature distillation with domain adaptation**: Match only the domain-invariant features, allowing the student to develop its own domain-specific features
4. **Fine-tuning**: After feature distillation, fine-tune the student on the target domain data

The Gram matrix matching approach is particularly well-suited for cross-domain transfer because it captures structural relationships between features rather than exact activation patterns. Two markets may have different price scales and volatility levels, but the correlational structure between trend features and volatility features may be similar.

A practical example: distilling features from a teacher trained on S&P 500 constituents to a student trading cryptocurrency futures. The universal features (momentum detection, mean reversion signals, volatility regime classification) transfer well, while the student learns crypto-specific features (funding rate dynamics, perpetual-spot basis, liquidation cascades) from the target domain data.

## 6. Implementation Walkthrough

Our Rust implementation provides a complete feature distillation framework with the following components:

### 6.1 Network Architecture

The teacher and student networks are multi-layer perceptrons with accessible intermediate features. The teacher has larger hidden dimensions and more layers, while the student is compact. Both networks expose their intermediate activations for feature matching.

```rust
// Teacher network with 3 hidden layers, exposing intermediate features
let teacher = TeacherNetwork::new(input_dim, &[128, 64, 32], output_dim);
let (output, features) = teacher.forward_with_features(&input);
// features: Vec<Array2<f64>> containing activations from each hidden layer

// Student network with 2 hidden layers, smaller dimensions
let student = StudentNetwork::new(input_dim, &[32, 16], output_dim);
let (output, features) = student.forward_with_features(&input);
```

### 6.2 Feature Distillation Losses

Each loss function operates on matched pairs of teacher and student feature maps:

```rust
// FitNet hint loss with projection
let hint_loss = fitnet_loss(&teacher_features[1], &student_features[0], &projection);

// Attention transfer loss
let at_loss = attention_transfer_loss(&teacher_features, &student_features);

// Gram matrix matching
let gram_loss = gram_matrix_loss(&teacher_features[1], &student_features[0]);
```

### 6.3 Combined Training

The training loop combines task loss, output distillation, and feature matching:

```rust
let config = DistillationConfig {
    alpha: 0.3,        // task loss weight
    beta: 0.3,         // KD loss weight
    gamma: vec![0.2, 0.2], // feature matching weights per layer
    temperature: 4.0,
    feature_method: FeatureMethod::Combined, // FitNet + Attention + Gram
};
```

### 6.4 Feature Analysis

After training, we analyze how well the student's features align with the teacher's:

```rust
let similarity = feature_similarity(&teacher_features, &student_features);
// Returns cosine similarity, CKA (Centered Kernel Alignment), and correlation metrics
```

## 7. Bybit Data Integration

The implementation fetches real market data from the Bybit exchange API for training and evaluation. We use the public klines (candlestick) endpoint:

```
GET https://api.bybit.com/v5/market/kline?category=linear&symbol=BTCUSDT&interval=5&limit=200
```

The data is preprocessed into features suitable for our models:
- **Price features**: normalized OHLC values, returns, log returns
- **Volume features**: normalized volume, volume-weighted price
- **Volatility features**: rolling standard deviation, range (high-low)
- **Technical indicators**: simple moving averages, RSI approximation

These features are computed from the raw kline data and used as input to both teacher and student models. The prediction target is the direction of the next candle's close relative to its open.

## 8. Key Takeaways

1. **Feature distillation transfers internal representations**, not just output behavior. This gives the student a much richer understanding of the input data and often leads to better generalization.

2. **Multiple complementary methods exist**: FitNets match raw activations, attention transfer matches spatial focus patterns, Gram matrix matching captures feature correlations, and CRD uses contrastive learning. Combining them often yields the best results.

3. **Layer selection and projection design are critical**. Matching too many layers can over-constrain the student; matching too few wastes the teacher's representational knowledge. Projection heads must be carefully sized to avoid information bottlenecks.

4. **Loss weighting requires careful tuning**. Feature matching losses should guide the student's representations without dominating the task-specific learning objective. Annealing feature loss weights during training is a practical strategy.

5. **Trading applications are natural**: market microstructure features are hierarchical, making them well-suited for feature distillation. Cross-timeframe and cross-asset transfer are particularly valuable for deploying compact models in latency-sensitive trading environments.

6. **Cross-domain transfer leverages universal market dynamics**. Structural features like momentum, mean reversion, and volatility clustering appear across markets, and Gram matrix matching is well-suited for transferring these structural relationships.

7. **Rust implementation enables production deployment**. The compiled, zero-overhead nature of Rust makes feature-distilled models suitable for low-latency trading systems where every microsecond counts.

8. **Feature analysis provides interpretability**. By examining which teacher features the student successfully learns, we gain insight into which market dynamics are captured by the distilled model, aiding in model validation and risk management.
