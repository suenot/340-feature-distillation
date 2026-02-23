use anyhow::Result;
use feature_distillation::*;

fn main() -> Result<()> {
    println!("=== Chapter 210: Feature Distillation for Trading ===\n");

    // -------------------------------------------------------------------------
    // Step 1: Fetch data from Bybit
    // -------------------------------------------------------------------------
    println!("Step 1: Fetching BTCUSDT kline data from Bybit...");
    let candles = match fetch_bybit_klines("BTCUSDT", "5", 200) {
        Ok(c) => {
            println!("  Fetched {} candles", c.len());
            c
        }
        Err(e) => {
            println!("  Warning: Could not fetch from Bybit ({}), using synthetic data", e);
            generate_synthetic_candles(200)
        }
    };

    let lookback = 5;
    let (features, labels) = candles_to_features(&candles, lookback);
    println!(
        "  Feature matrix: {} samples x {} features",
        features.nrows(),
        features.ncols()
    );
    println!(
        "  Label distribution: {:.1}% positive\n",
        labels.iter().filter(|&&l| l > 0.5).count() as f64 / labels.len() as f64 * 100.0
    );

    // Split into train/test
    let split = (features.nrows() as f64 * 0.7) as usize;
    let train_x = features.slice(ndarray::s![..split, ..]).to_owned();
    let train_y = labels.slice(ndarray::s![..split]).to_owned();
    let test_x = features.slice(ndarray::s![split.., ..]).to_owned();
    let test_y = labels.slice(ndarray::s![split..]).to_owned();
    println!(
        "  Train: {} samples, Test: {} samples\n",
        train_x.nrows(),
        test_x.nrows()
    );

    let input_dim = features.ncols();
    let teacher_hidden = vec![128, 64, 32];
    let student_hidden = vec![32, 16];
    let epochs = 200;
    let lr = 0.01;

    // -------------------------------------------------------------------------
    // Step 2: Train the teacher model
    // -------------------------------------------------------------------------
    println!("Step 2: Training teacher model (hidden: {:?})...", teacher_hidden);
    let mut teacher = TeacherNetwork::new(input_dim, &teacher_hidden, 1);
    for epoch in 0..epochs {
        let loss = teacher.train_step(&train_x, &train_y, lr);
        if epoch % 50 == 0 || epoch == epochs - 1 {
            let train_acc = compute_accuracy(&teacher.forward(&train_x), &train_y);
            let test_acc = compute_accuracy(&teacher.forward(&test_x), &test_y);
            println!(
                "  Epoch {}: loss={:.6}, train_acc={:.2}%, test_acc={:.2}%",
                epoch,
                loss,
                train_acc * 100.0,
                test_acc * 100.0
            );
        }
    }

    let teacher_train_acc = compute_accuracy(&teacher.forward(&train_x), &train_y);
    let teacher_test_acc = compute_accuracy(&teacher.forward(&test_x), &test_y);
    println!(
        "\nTeacher final: train_acc={:.2}%, test_acc={:.2}%\n",
        teacher_train_acc * 100.0,
        teacher_test_acc * 100.0
    );

    // Extract teacher features for analysis
    let (_, teacher_test_features) = teacher.forward_with_features(&test_x);

    // -------------------------------------------------------------------------
    // Step 3: Train student WITHOUT distillation (baseline)
    // -------------------------------------------------------------------------
    println!("Step 3: Training student WITHOUT distillation (hidden: {:?})...", student_hidden);
    let mut student_baseline = StudentNetwork::new(input_dim, &student_hidden, 1);
    for epoch in 0..epochs {
        let loss = student_baseline.train_step(&train_x, &train_y, lr);
        if epoch % 50 == 0 || epoch == epochs - 1 {
            let test_acc = compute_accuracy(&student_baseline.forward(&test_x), &test_y);
            println!(
                "  Epoch {}: loss={:.6}, test_acc={:.2}%",
                epoch,
                loss,
                test_acc * 100.0
            );
        }
    }

    let baseline_train_acc = compute_accuracy(&student_baseline.forward(&train_x), &train_y);
    let baseline_test_acc = compute_accuracy(&student_baseline.forward(&test_x), &test_y);
    println!(
        "\nBaseline student final: train_acc={:.2}%, test_acc={:.2}%\n",
        baseline_train_acc * 100.0,
        baseline_test_acc * 100.0
    );

    // -------------------------------------------------------------------------
    // Step 4: Train student with standard output distillation
    // -------------------------------------------------------------------------
    println!("Step 4: Training student with standard output distillation...");
    let config_output_only = DistillationConfig {
        alpha: 0.5,
        beta: 0.5,
        gamma: vec![0.0, 0.0], // No feature matching
        temperature: 4.0,
        feature_method: FeatureMethod::FitNet,
        learning_rate: lr,
        epochs,
    };

    let trainer_output = FeatureDistillationTrainer::new(
        config_output_only,
        &teacher_hidden,
        &student_hidden,
    );
    let mut student_output_kd = StudentNetwork::new(input_dim, &student_hidden, 1);
    trainer_output.train(&teacher, &mut student_output_kd, &train_x, &train_y);

    let output_kd_train_acc =
        compute_accuracy(&student_output_kd.forward(&train_x), &train_y);
    let output_kd_test_acc =
        compute_accuracy(&student_output_kd.forward(&test_x), &test_y);
    println!(
        "\nOutput KD student final: train_acc={:.2}%, test_acc={:.2}%\n",
        output_kd_train_acc * 100.0,
        output_kd_test_acc * 100.0
    );

    // -------------------------------------------------------------------------
    // Step 5: Train student with FEATURE distillation (FitNet)
    // -------------------------------------------------------------------------
    println!("Step 5: Training student with FitNet feature distillation...");
    let config_fitnet = DistillationConfig {
        alpha: 0.3,
        beta: 0.3,
        gamma: vec![0.2, 0.2],
        temperature: 4.0,
        feature_method: FeatureMethod::FitNet,
        learning_rate: lr,
        epochs,
    };

    let trainer_fitnet = FeatureDistillationTrainer::new(
        config_fitnet,
        &teacher_hidden,
        &student_hidden,
    );
    let mut student_fitnet = StudentNetwork::new(input_dim, &student_hidden, 1);
    trainer_fitnet.train(&teacher, &mut student_fitnet, &train_x, &train_y);

    let fitnet_train_acc = compute_accuracy(&student_fitnet.forward(&train_x), &train_y);
    let fitnet_test_acc = compute_accuracy(&student_fitnet.forward(&test_x), &test_y);
    println!(
        "\nFitNet student final: train_acc={:.2}%, test_acc={:.2}%\n",
        fitnet_train_acc * 100.0,
        fitnet_test_acc * 100.0
    );

    // -------------------------------------------------------------------------
    // Step 6: Train student with COMBINED feature distillation
    // -------------------------------------------------------------------------
    println!("Step 6: Training student with Combined feature distillation (FitNet + AT + Gram)...");
    let config_combined = DistillationConfig {
        alpha: 0.3,
        beta: 0.2,
        gamma: vec![0.25, 0.25],
        temperature: 4.0,
        feature_method: FeatureMethod::Combined,
        learning_rate: lr,
        epochs,
    };

    let trainer_combined = FeatureDistillationTrainer::new(
        config_combined,
        &teacher_hidden,
        &student_hidden,
    );
    let mut student_combined = StudentNetwork::new(input_dim, &student_hidden, 1);
    trainer_combined.train(&teacher, &mut student_combined, &train_x, &train_y);

    let combined_train_acc =
        compute_accuracy(&student_combined.forward(&train_x), &train_y);
    let combined_test_acc =
        compute_accuracy(&student_combined.forward(&test_x), &test_y);
    println!(
        "\nCombined student final: train_acc={:.2}%, test_acc={:.2}%\n",
        combined_train_acc * 100.0,
        combined_test_acc * 100.0
    );

    // -------------------------------------------------------------------------
    // Step 7: Compare all models
    // -------------------------------------------------------------------------
    println!("========================================");
    println!("         COMPARISON SUMMARY");
    println!("========================================");
    println!(
        "Teacher ({}params):           train={:.2}%, test={:.2}%",
        count_teacher_params(&teacher),
        teacher_train_acc * 100.0,
        teacher_test_acc * 100.0
    );
    println!(
        "Student baseline:             train={:.2}%, test={:.2}%",
        baseline_train_acc * 100.0,
        baseline_test_acc * 100.0
    );
    println!(
        "Student output KD:            train={:.2}%, test={:.2}%",
        output_kd_train_acc * 100.0,
        output_kd_test_acc * 100.0
    );
    println!(
        "Student FitNet distillation:   train={:.2}%, test={:.2}%",
        fitnet_train_acc * 100.0,
        fitnet_test_acc * 100.0
    );
    println!(
        "Student Combined distillation: train={:.2}%, test={:.2}%",
        combined_train_acc * 100.0,
        combined_test_acc * 100.0
    );
    println!("========================================\n");

    // -------------------------------------------------------------------------
    // Step 8: Feature similarity analysis
    // -------------------------------------------------------------------------
    println!("Step 8: Feature similarity analysis (teacher vs each student)...\n");

    let models: Vec<(&str, &StudentNetwork)> = vec![
        ("Baseline", &student_baseline),
        ("Output KD", &student_output_kd),
        ("FitNet", &student_fitnet),
        ("Combined", &student_combined),
    ];

    for (name, student) in &models {
        println!("  {} student:", name);
        let (_, student_test_features) = student.forward_with_features(&test_x);
        let reports = analyze_feature_similarity(&teacher_test_features, &student_test_features);
        for report in &reports {
            println!("    {}", report);
        }
        println!();
    }

    println!("=== Feature Distillation experiment complete ===");
    Ok(())
}

/// Generate synthetic candle data when Bybit API is unavailable
fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut price = 50000.0_f64;
    let mut candles = Vec::with_capacity(n);

    for i in 0..n {
        let change = rng.gen_range(-0.02..0.02);
        let open = price;
        let close = price * (1.0 + change);
        let high = open.max(close) * (1.0 + rng.gen_range(0.0..0.01));
        let low = open.min(close) * (1.0 - rng.gen_range(0.0..0.01));
        let volume = rng.gen_range(100.0..10000.0);

        candles.push(Candle {
            timestamp: 1000000 + (i as u64 * 300),
            open,
            high,
            low,
            close,
            volume,
        });

        price = close;
    }

    candles
}

/// Count approximate number of parameters in teacher network
fn count_teacher_params(teacher: &TeacherNetwork) -> usize {
    let mut count = 0;
    for layer in &teacher.layers {
        count += layer.weights.len() + layer.bias.len();
    }
    count += teacher.output_layer.weights.len() + teacher.output_layer.bias.len();
    count
}
