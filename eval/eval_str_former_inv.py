"""
Only for evaluating the strformer
"""

import pickle

import numpy as np
import torch
import wandb

from lawt.envs.numeric import NumericEnvironment
from lawt.model.transformer import TransformerModel
from eval.eval_utils import MATRIX_LIST, construct_matrix

# Laplace
model_args_inv = pickle.load(open("./saved_runs/inverse/params.pkl", mode="rb"))
env_inv = NumericEnvironment(model_args_inv)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_inv = torch.load("./saved_runs/inverse/weights.pth", map_location=device)
encoder_inv = TransformerModel(model_args_inv, env_inv.id2word, is_encoder=True, with_output=False)
decoder_inv = TransformerModel(model_args_inv, env_inv.id2word, is_encoder=False, with_output=True)

encoder_inv.load_state_dict(checkpoint_inv["encoder"])
decoder_inv.load_state_dict(checkpoint_inv["decoder"])

encoder_inv.eval()
decoder_inv.eval()


def generate_tokens(encoder, decoder, env, matrix):
    x_symbols = env.input_encoder.encode(matrix)
    x_ids = [env.word2id[s] for s in x_symbols]
    x_tensor = torch.LongTensor(x_ids).unsqueeze(1).to(device)  # (slen, 1)
    lengths = torch.LongTensor([x_tensor.size(0)]).to(device)

    # 2) encode once
    with torch.no_grad():
        enc = encoder.fwd(x=x_tensor, lengths=lengths, causal=False)
        src_enc = enc.transpose(0, 1)  # now (batch=1, slen, dim)

        # 3) call the built-in generate
        max_len = env.max_output_length + 2
        # returns generated tokens (slen, batch) and optional scores
        generated, _ = decoder.generate(src_enc, lengths, max_len=max_len)

        generated = generated.transpose(0, 1)
        gen_ids = generated[0].tolist()

    # 5) map back to words
    return [env.id2word[i] for i in gen_ids]


def predict_w_decoder(encoder, decoder, env, matrix):
    pred_symbols = generate_tokens(encoder, decoder, env, matrix)

    decoded = env.output_encoder.decode(pred_symbols[1:-1])

    if decoded is None:
        print("Decode failed")
        print(pred_symbols)
        return None

    decoded = decoded.squeeze()

    return decoded


def run_eval(encoder, decoder, env, model_name):
    wandb.init(project="trans_nla_eval", name=model_name)
    matrix_sizes = [50]
    for matrix_size in matrix_sizes:
        for matrix_type in MATRIX_LIST:
            targets = []
            preds = []

            for _ in range(0, 15):
                matrix = construct_matrix(matrix_size, matrix_type, bsz=1)

                matrix = matrix.squeeze()

                decoded = predict_w_decoder(encoder, decoder, env, matrix)

                if decoded is None:
                    print("Decode failed")
                    continue

                if model_name == "Inverse_50":
                    pred = decoded

                    if pred.shape != matrix.shape:
                        continue
                    else:
                        pred_matr = pred @ np.array(matrix)
                        preds.append(pred_matr)

            if model_name == "Inverse_50":
                targets = np.array([np.eye(matrix_size) for _ in range(len(preds))])

                preds = np.array(preds)

            if len(preds) == 0:
                wandb.summary[f"support_{matrix_type}_{matrix_size}"] = 0
                continue

            axis = tuple(range(1, targets.ndim))

            norms = [1, 2, "fro", "nuc"]

            test_rel_err_means = {}

            for norm in norms:
                test_rel_errors = np.linalg.norm(targets - preds, axis=axis, ord=norm) / np.linalg.norm(
                    targets, axis=axis, ord=norm
                )

                test_rel_err_means[f"{matrix_size}_{norm}"] = np.mean(test_rel_errors)

            bsz = len(targets)

            sl = (
                (slice(None, 2),) + (slice(None, 3),) * (targets.ndim - 1) if targets.ndim > 2 else (slice(None, 10), 0)
            )

            print(f"{model_name} | Matrix Type: {matrix_type} | bsz: {bsz} | {matrix_size}x{matrix_size}")
            print(f"Targets: {targets[sl]}")
            print(f"Preds: {preds[sl]}")

            wandb.log({"targets": targets[sl], "preds": preds[sl]})

            for key, value in test_rel_err_means.items():
                print(f"{key}: {value}")
                wandb.summary[f"{matrix_type}_{key}_{matrix_size}"] = value

            wandb.summary[f"support_{matrix_type}_{matrix_size}"] = bsz
            print("-" * 100)
            print("\n")

    wandb.finish()


run_eval(encoder_inv, decoder_inv, env_inv, "Inverse_50")
