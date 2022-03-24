def build_run_name(config):
    run_name = config.model_name
    run_name += '_' + config.dataset_name
    run_name += '_' + config.model_type
    run_name += '_' + config.optimizer
    run_name += '_' + config.loss
    run_name += '_' + config.lr_scheduler
    run_name += '_' + config.batch_size
    run_name += '_' + config.num_workers
    run_name += '_' + config.num_epochs
    run_name += '_' + config.seed
    return run_name