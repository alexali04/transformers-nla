import argparse


def get_training_parser():
    parser = argparse.ArgumentParser(description="Transformers NLA")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--data_path", nargs="+", type=str, default=["./datasets"])
    parser.add_argument("--bsz", type=int, default=1024)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--sch_name", type=str, default="cons")
    parser.add_argument("--sch_sz", type=int, default=1_000)
    parser.add_argument("--sch_gamma", type=float, default=0.9)
    parser.add_argument("--sch_thres", type=float, default=1e-8)
    parser.add_argument("--sch_batches", type=int, default=10)
    parser.add_argument("--sch_sigma_thres", type=float, default=0.9)
    parser.add_argument("--grad_norm", type=float, default=1e6)
    parser.add_argument("--model", type=str, default="num_trans")
    parser.add_argument("--loss", type=str, default="l1")
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--log_dir", type=str, default="./saved_runs")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb_proj", type=str, default="rebuttal_auto_nla")
    parser.add_argument("--prop_log", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--pos_enc", type=str, default="learned")
    parser.add_argument("--attn_pattern", type=str, default="linear_taylor")
    parser.add_argument("--dim_proportion", type=float, default=0.05)
    parser.add_argument("--max_seq", type=int)
    parser.add_argument("--alpha", type=float, default=0.98)
    parser.add_argument("--lambda", type=float, default=2.0)
    parser.add_argument("--bias", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--pres", type=str, default="single")
    parser.add_argument("--input_inj", action="store_true")
    parser.add_argument("--sample_rec", type=str, default="cons")
    parser.add_argument("--k_layer_loop", type=int, default=1)
    parser.add_argument("--mlp_n_layer", type=int, default=2)
    parser.add_argument("--mlp_factor", type=int, default=4)
    parser.add_argument("--mlp_act", type=str, default="relu")
    parser.add_argument("--out_dim", type=int)
    parser.add_argument("--channel_count", type=int, default=1)
    parser.add_argument("--target_size", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=2)
    parser.add_argument("--eval_conv_sizes", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--eval_conv_bsz", type=int, default=100)

    return parser

def clean_training_parser(args):
    if any(case in args.data_path[0] for case in ["largest_eigenvalue", "trace", "log_det"]):
        setattr(args, "out_dim", 1)
    return args


def get_data_parser():
    parser = argparse.ArgumentParser(description="Transformers NLA Data")
    parser.add_argument("--op", type=str, default="largest_eigenvalue")
    parser.add_argument("--distribution", type=str, default="gaussian")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--N", type=int, default=7)
    parser.add_argument("--B", type=int, default=13)
    parser.add_argument("--variances", nargs="+", type=float, default=[1.0])
    parser.add_argument("--dists", nargs="+", type=str, default=["gaussian"])
    parser.add_argument("--struct_n", type=int, default=20)
    parser.add_argument("--rank_pct", type=float, default=-1)
    parser.add_argument("--eps", type=float, default=1e-4)

    return parser


def get_eval_parser():
    parser = argparse.ArgumentParser(description="Transformers NLA Eval")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--eval_sizes", nargs="+", type=int)
    parser.add_argument("--eval_bsz", type=int, default=20)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb_proj", type=str, default="trans_nla_eval")
    parser.add_argument("--it_eval", action=argparse.BooleanOptionalAction)
    parser.add_argument("--laser", action=argparse.BooleanOptionalAction)
    parser.add_argument("--laser_prop", type=float)
    parser.add_argument("--laser_strategy", type=str, default="mlp")
    return parser


def clean_eval_parser(args):
    if not args.laser:
        for case in ["laser_strategy", "laser_prop"]:
            setattr(args, "laser_strategy", None)
    return args


def get_gp_parser():
    parser = get_training_parser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--vtol", type=float, default=0.2)
    parser.add_argument("--case", type=str, default="net")
    parser.set_defaults(wandb_proj="gp_nla")
    return parser


def get_krylov_parser():
    parser = get_training_parser()
    parser.add_argument("--model_path", type=str, default="")
    parser.set_defaults(seed=21)
    parser.add_argument("--alg", type=str, default="CG")
    parser.add_argument("--case", type=str, default="rbf")
    parser.add_argument("--trans", type=str, default="slice")
    parser.add_argument("--precision", type=str, default="single")
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.set_defaults(log_dir="logs")
    return parser
