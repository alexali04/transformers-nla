"""
Only for evaluating the strformer
"""

import pickle

import numpy as np
import torch
import wandb

from lawt.envs.numeric import NumericEnvironment
from lawt.model.transformer import TransformerModel
from eval.eval_utils import construct_matrix

MATRIX_LIST = ["Toeplitz", "MM Slice", "MM Dim"]

# Laplace
model_args_inv = pickle.load(open("./saved_runs/eigenvalue_laplace/params.pkl", mode="rb"))
env_inv = NumericEnvironment(model_args_inv)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_inv = torch.load("./saved_runs/eigenvalue_laplace/weights.pth", map_location=device)
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
    matrix_sizes = [5, 10, 20, 50]
    for matrix_size in matrix_sizes:
        for matrix_type in MATRIX_LIST:
            targets = []
            preds = []

            for _ in range(0, 15):
                matrix = construct_matrix(matrix_size, matrix_type, bsz=1, psd=True)

                matrix = matrix.squeeze()

                decoded = predict_w_decoder(encoder, decoder, env, matrix)

                matrix = matrix.numpy()

                assert np.all(np.linalg.eigvalsh(matrix) > 0), "Matrix must be pos-def"

                true_logdet = np.linalg.slogdet(matrix)[1]

                if decoded is None:
                    print("Decode failed")
                    continue

                num_neg_eigvals = len(decoded[decoded < 0])

                if num_neg_eigvals % 2 == 1:
                    print("Odd number of negative eigenvalues")
                    continue

                pos_eigvals = decoded[decoded > 0]

                pos_log_det = np.sum(np.log(pos_eigvals))

                neg_eigvals = np.prod(decoded[decoded < 0])

                pred = pos_log_det + np.log(neg_eigvals)

                targets.append(true_logdet)
                preds.append(pred)

            targets = np.array(targets)
            preds = np.array(preds)

            rel_errs = np.abs(targets - preds) / np.abs(np.where(targets < 1e-7, 1.0, targets))
            rel_err_mean = np.mean(rel_errs)

            bsz = len(targets)

            # sl = (
            #     (slice(None, 2),) + (slice(None, 3),) * (targets.ndim - 1) if targets.ndim > 2 else (slice(None, 10), 0)
            # )

            print(f"{model_name} | Matrix Type: {matrix_type} | bsz: {bsz} | {matrix_size}x{matrix_size}")
            print(f"Targets: {targets[: min(bsz, 10)]}")
            print(f"Preds: {preds[: min(bsz, 10)]}")
            print(f"Mean Relative Error: {rel_err_mean}")

            wandb.summary[f"{matrix_type}_{matrix_size}"] = rel_err_mean
            wandb.summary[f"support_{matrix_type}_{matrix_size}"] = bsz

            print("-" * 100)
            print("\n")

    wandb.finish()


run_eval(encoder_inv, decoder_inv, env_inv, "LogDet")
