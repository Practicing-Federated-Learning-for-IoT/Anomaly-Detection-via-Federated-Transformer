import argparse
DATASET_PATH = r'F:\Code\Anomaly Detection\FedAnomaly\data\\'


class Parser:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def set_arguments(self):
        self.parser.add_argument('-path', type=str, default= DATASET_PATH)
        self.parser.add_argument('-d', '--dataset', type=str)
        self.parser.add_argument('-c', '--client', type=int)
        self.parser.add_argument('-f', '--frac', type=float, help='to set fraction of clients per round')
        self.parser.add_argument('-b', '--batch_size',default=32, type=int)
        self.parser.add_argument('-d_data',  type=int)
        self.parser.add_argument('-d_feature', type=int, default=16)
        self.parser.add_argument('-heads', type=int)
        self.parser.add_argument('-r','--radio',type=float)
        self.parser.add_argument('-e', '--epoch',type=int)


    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args
