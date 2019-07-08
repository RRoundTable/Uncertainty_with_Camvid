def get_hyperparams():
    hyperprams = {
        "dataset_path": "",
        "batch_size": 1,
        "image_shape": [360, 480],
        "num_classes": 12,
        "learning_rate": 1e-4,
        "dropout": 0.5,
        "lr_decay": 0.995,
        "decay_per_n_epoch": 1,
        "n_epoch": 20,
        "model": "FCDenseNet103",
        "mode": "combined",
    }
    return hyperprams
