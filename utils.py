from argparse import ArgumentParser

from config import ParallelSpeechAndTextModelConfig


def construct_arguments_parser():
  parser = ArgumentParser()
  parser.add_argument("--pretraining_contrastive_loss", type=str, default="TripletMarginLoss", help="TripelMarginLoss or SimCLR")
  parser.add_argument("--train_last_n_layers", type=int, default=0)
  parser.add_argument("--training_mode", type=str, default="pretrain", help="pretrain or finetune")
  parser.add_argument("--train_batch_size", type=int, default=64)
  parser.add_argument("--val_batch_size", type=int, default=512)
  parser.add_argument("--num_epochs", type=int, default=10)
  parser.add_argument("--dataset_name", type=str, default="librispeech", help="librispeech or spotify")
  return parser


def build_config_from_args(parser):
  #TODO have different configs for model, data, trainer?
  #TODO build corresponding config class depending on user input, options: bi-encoder, multimodal encoder
  args = parser.parse_args()
  config = ParallelSpeechAndTextModelConfig(pretraining_contrastive_loss_fn=args.pretraining_contrastive_loss,
                                            train_last_n_layers=args.train_last_n_layers,
                                            training_mode=args.training_mode,
                                            train_batch_size=args.train_batch_size,
                                            val_batch_size=args.val_batch_size,
                                            num_epochs=args.num_epochs,
                                            dataset_name=args.dataset_name)
  return config


def build_run_name_from_config(config):
  run_name = config.model_name
  run_name += '_' + config.training_mode
  run_name += '_' + config.dataset_name
  run_name += '_los-fn-' + config.pretraining_contrastive_loss_fn
  run_name += '_train-batch-' + str(config.train_batch_size)
  run_name += '_val-batch-' + str(config.val_batch_size)
  run_name += '_epochs-' + str(config.num_epochs)
  return run_name


def build_model_from_config(config):
  pass


def build_data_module_from_config(config):
  pass
