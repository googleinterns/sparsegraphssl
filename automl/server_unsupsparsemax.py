from automl.full_eval import start
from automl.full_eval import get_default_parser

if __name__ == '__main__':
    parser = get_default_parser()
    args = parser.parse_args()
    args.shared_directory = '/mnt/datasets/idea/ssl/unsupsparsemax'
    args.min_budget = 400
    args.max_budget = 400
    args.automl_worker = 'worker_unsupsparsemax'
    # print(args.gpu)
    start(args)