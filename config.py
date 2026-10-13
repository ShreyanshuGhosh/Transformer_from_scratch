from pathlib import Path

def get_config():
    return{
        "batch_size" :4,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "datasource": "opus_books",
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizers/{0}.json",        
        "d_model": 512,
        "experiment_name": "runs/tmodel",

    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path('.')/ model_folder/model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    weights_files = sorted(Path(model_folder).glob(f"{config['model_basename']}*.pt"))
    if not weights_files:
        return None
    return str(weights_files[-1])

