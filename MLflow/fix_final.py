import json

with open(r'C:\Users\aili\Desktop\MLflow\mlflow.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ===== 1. CatBoost function: добавляем train/val метрики =====
new_catboost_fn = r"""def train_catboost(X_train, y_train, X_val, y_val, X_test, y_test):

    print(f"\n{BLUE}Обучаем CatBoost с оптимизацией гиперпараметров...{RESET}")

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_seed': 42,
            'verbose': False
        }

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        return r2_score(y_val, model.predict(X_val))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    best_params = study.best_params
    catboost_model = CatBoostRegressor(**best_params, random_seed=42, verbose=False)
    catboost_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    y_pred_test = catboost_model.predict(X_test)
    y_pred_train = catboost_model.predict(X_train)
    y_pred_val = catboost_model.predict(X_val)

    results = {
        'model': catboost_model,
        'best_params': best_params,
        'r2': r2_score(y_test, y_pred_test),
        'mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'val_r2': r2_score(y_val, y_pred_val),
        'val_mae': mean_absolute_error(y_val, y_pred_val),
        'predictions': y_pred_test
    }

    print(f"Лучшие параметры: iterations={best_params['iterations']}, depth={best_params['depth']}, lr={best_params['learning_rate']:.3f}")
    print(f"R^2: {results['r2']:.4f}")

    return results"""

# ===== 2. CatBoost run cell: добавляем train/val в log_metrics =====
new_catboost_run = r"""experiment_name = "gradient_boosting_optimized"
artifact_location = "s3://mlflow-bucket/mlflow"

exp = mlflow.get_experiment_by_name(experiment_name)
if exp is None:
    exp_id = MlflowClient().create_experiment(name=experiment_name, artifact_location=artifact_location)
else:
    exp_id = exp.experiment_id

mlflow.set_experiment(experiment_name)
registered_model_name = "catboost_optimized"

with mlflow.start_run(run_name="catboost_optimized_final", experiment_id=exp_id):
    catboost_results = train_catboost(X_train, y_train, X_val, y_val, X_test, y_test)

    mlflow.log_metrics({
        "test_r2": catboost_results['r2'],
        "test_mae": catboost_results['mae'],
        "train_r2": catboost_results['train_r2'],
        "train_mae": catboost_results['train_mae'],
        "val_r2": catboost_results['val_r2'],
        "val_mae": catboost_results['val_mae'],
    })
    mlflow.log_params(catboost_results['best_params'])

    signature = infer_signature(X_train, catboost_results['model'].predict(X_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=catboost_results['model'],
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5),
        registered_model_name=registered_model_name,
    )

    client = MlflowClient()
    client.set_model_version_tag(registered_model_name, model_info.registered_model_version, "env", "PRD")
    client.set_registered_model_alias(registered_model_name, "prd", model_info.registered_model_version)

    print(f"Модель зарегистрирована: {registered_model_name} v{model_info.registered_model_version}")
    print("Alias 'prd' и тег PRD добавлены")"""

# ===== 3. Robustness: фиксируем вывод на реальных данных =====
new_robustness = r"""with mlflow.start_run(run_name="robustness_best_model", nested=True):
    noise_levels = [0.01, 0.05, 0.1]
    X_sample = X_test.iloc[:100]
    y_sample = y_test.iloc[:100]

    original_pred = best_model.predict(X_sample)
    original_mse = mean_squared_error(y_sample, original_pred)
    mlflow.log_metric("baseline_mse", original_mse)

    print(f"\n{BLUE}ROBUSTNESS TEST ({best_model_name}):{RESET}")

    change_pcts = []
    for noise in noise_levels:
        X_noisy = X_sample.copy()
        numeric_cols = X_noisy.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            noise_vals = np.random.normal(0, noise * X_noisy[col].std(), len(X_noisy))
            X_noisy[col] += noise_vals

        noisy_pred = best_model.predict(X_noisy)
        noisy_mse = mean_squared_error(y_sample, noisy_pred)
        change_pct = ((noisy_mse - original_mse) / original_mse) * 100
        change_pcts.append(change_pct)

        mlflow.log_metric(f"mse_noise_{int(noise*100)}", noisy_mse)
        mlflow.log_metric(f"mse_change_{int(noise*100)}", change_pct)

        status = "устойчива" if abs(change_pct) < 10 else "чувствительна"
        print(f"  Noise {int(noise*100):2d}%: изменение MSE = {change_pct:+6.1f}% -> {status}")

    print(f"\n{BLUE}ТАКИМ ОБРАЗОМ:{RESET}")
    if all(abs(cp) < 10 for cp in change_pcts):
        print("Модель демонстрирует хорошую устойчивость к шуму до 10%")
    elif abs(change_pcts[0]) < 10:
        print("Модель устойчива к малому шуму (1-5%), но чувствительна к сильному шуму (>10%)")
    else:
        print("Модель чувствительна к шуму на всех уровнях")"""

# Применяем изменения
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])

    if 'def train_catboost' in src:
        cell['source'] = [new_catboost_fn]
        cell['outputs'] = []
        print("CatBoost function updated.")

    elif 'catboost_optimized_final' in src:
        cell['source'] = [new_catboost_run]
        cell['outputs'] = []
        print("CatBoost run cell updated.")

    elif 'ТАКИМ ОБРАЗОМ' in src and 'robustness' in src.lower():
        cell['source'] = [new_robustness]
        cell['outputs'] = []
        print("Robustness cell updated.")

with open(r'C:\Users\aili\Desktop\MLflow\mlflow.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Done.")
