import argparse
from Training.train_search import *
from Training.train_tracing import *
from Training.train_full import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_model', default='full', type=str, help='model to train: search, trace, or full')
    parser.add_argument('--target_selection', default=True, type=bool, help='select target with agent')
    parser.add_argument('--search_weights', default=None, type=str, help='weights to load for search model')
    parser.add_argument('--trace_weights', default=None, type=str, help='weights to load for trace model')
    parser.add_argument('--target_selection_weights', default=None, type=str, help='weights to load for selection model')
    args = parser.parse_args()

    if args.train_model == 'search':
        train_search_agent(args.search_weights)

    elif args.train_model == 'trace':
        train_tracing_agent(args.trace_weights)

    elif args.train_model == 'full':
        train_full_model(not args.target_selection, args.search_weights, args.trace_weights, args.target_selection_weights)
