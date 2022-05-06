'''
Main Sweep
'''

# Imports
import json
import argparse
import functools
from pprint import pprint

from Model import *

# Main Functions
# Wandb Sweep Function
def Model_Sweep_Run(wandb_data):
    '''
    Model Sweep Runner
    '''
    # Init
    wandb.init()

    # Get Run Config
    config = wandb.config
    N_EPOCHS = config.n_epochs
    BATCH_SIZE = config.batch_size

    ENCODER = config.encoder
    DECODER = config.decoder
    ENCODER_EMBEDDING_SIZE = config.encoder_embedding_size
    DECODER_EMBEDDING_SIZE = config.decoder_embedding_size
    ENCODER_N_UNITS = config.encoder_n_units
    DECODER_N_UNITS = config.decoder_n_units
    ACT_FUNC = config.act_func
    DROPOUT = config.dropout
    USE_ATTENTION = False

    print("RUN CONFIG:")
    pprint(config)

    # Get Inputs
    inputs = {
        "model": {
            "blocks": {
                "encoder": [
                    functools.partial(BLOCKS_ENCODER[ENCODER], 
                        n_units=ENCODER_N_UNITS[i], activation=ACT_FUNC, 
                        dropout=DROPOUT, recurrent_dropout=DROPOUT, 
                        return_state=True, return_sequences=(i < (len(ENCODER_N_UNITS)-1)), 
                    ) for i in range(len(ENCODER_N_UNITS))
                ],
                "decoder": [
                    functools.partial(BLOCKS_DECODER[DECODER], 
                        n_units=DECODER_N_UNITS[i], activation=ACT_FUNC, 
                        dropout=DROPOUT, recurrent_dropout=DROPOUT, 
                        return_state=True, return_sequences=True, 
                    ) for i in range(len(DECODER_N_UNITS))
                ],
            }, 
            "compile_params": {
                "loss_fn": CategoricalCrossentropy(),#SparseCategoricalCrossentropy(),
                "optimizer": Adam(),
                "metrics": ["accuracy"]
            }
        }
    }

    # Get Train Val Dataset
    DATASET, DATASET_ENCODED = LoadTrainDataset_Dakshina(
        DATASET_PATH_DAKSHINA_TAMIL
    )
    inputs["dataset_encoded"] = DATASET_ENCODED
    inputs["dataset_encoded"]["train"]["batch_size"] = BATCH_SIZE
    inputs["dataset_encoded"]["val"]["batch_size"] = BATCH_SIZE

    # Build Model
    X_shape = DATASET_ENCODED["train"]["encoder_input"].shape
    Y_shape = DATASET_ENCODED["train"]["decoder_output"].shape
    MODEL = Model_EncoderDecoderBlocks(
        X_shape=X_shape, Y_shape=Y_shape, 
        Blocks=inputs["model"]["blocks"],
        encoder={
            "embedding_size": ENCODER_EMBEDDING_SIZE
        }, 
        decoder={
            "embedding_size": DECODER_EMBEDDING_SIZE
        },
        use_attention=USE_ATTENTION
    )
    MODEL = Model_Compile(MODEL, **inputs["model"]["compile_params"])

    # Train Model
    TRAINED_MODEL, TRAIN_HISTORY = Model_Train(MODEL, inputs, N_EPOCHS, wandb_data, best_model_path=PATH_BESTMODEL)

    # Load Best Model
    TRAINED_MODEL = Model_LoadModel(PATH_BESTMODEL)
    # Get Test Dataset
    DATASET_TEST, DATASET_ENCODED_TEST = LoadTestDataset_Dakshina(
        DATASET_PATH_DAKSHINA_TAMIL
    )
    # Test Best Model
    loss_test, eval_test, eval_test_inference = Model_Test(
        TRAINED_MODEL, DATASET_ENCODED_TEST,
        target_chars=DATASET_ENCODED_TEST["chars"]["target_chars"],
        target_char_map=DATASET_ENCODED_TEST["chars"]["target_char_map"],
        use_attention=USE_ATTENTION
    )

    # Wandb log test data
    wandb.log({
        "loss_test": loss_test,
        "eval_test": eval_test,
        "eval_test_inference": eval_test_inference
    })

    # Close Wandb Run
    # run_name = "ep:"+str(N_EPOCHS) + "_" + "bs:"+str(BATCH_SIZE) + "_" + "nf:"+str(N_FILTERS) + "_" + str(DROPOUT)
    # wandb.run.name = run_name
    wandb.finish()

# Run