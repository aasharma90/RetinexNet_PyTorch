import os
import argparse
from glob import glob
import numpy as np
from model import RetinexNet

parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpu_id', dest='gpu_id', 
                    default="0",
                    help='GPU ID (-1 for CPU)')
parser.add_argument('--data_dir', dest='data_dir',
                    default='./data/test/low/',
                    help='directory storing the test data')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', 
                    default='./ckpts/',
                    help='directory for checkpoints')
parser.add_argument('--res_dir', dest='res_dir', 
                    default='./results/test/low/',
                    help='directory for saving the results')

args = parser.parse_args()

def predict(model):

    test_low_data_names  = glob(args.data_dir + '/' + '*.*')
    test_low_data_names.sort()
    print('Number of evaluation images: %d' % len(test_low_data_names))

    model.predict(test_low_data_names,
                res_dir=args.res_dir,
                ckpt_dir=args.ckpt_dir)


if __name__ == '__main__':
    if args.gpu_id != "-1":
        # Create directories for saving the results
        if not os.path.exists(args.res_dir):
            os.makedirs(args.res_dir)
        # Setup the CUDA env
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        # Create the model
        model = RetinexNet().cuda()
        # Test the model
        predict(model)
    else:
        # CPU mode not supported at the moment!
        raise NotImplementedError
