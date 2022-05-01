import flwr as fl
from multi.ServerStrategy.Fedavg import OurFed
from utils.options import args_parser
args = args_parser()
strategy = OurFed(
    min_fit_clients=int(args.num_users * args.frac),
    min_eval_clients=args.num_users,
    min_available_clients = args.num_users
)
fl.server.start_server(server_address=args.ip,config={"num_rounds": args.epochs},strategy=strategy)