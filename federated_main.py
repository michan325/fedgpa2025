import numpy as np
import torch
from data_loader import get_dataset
from running import one_round_training
from methods import local_update
from models import CifarCNN, CNN_FMNIST, ResNet18
from options import args_parser
import copy
import time


if __name__ == '__main__':

    start_time = time.time()

    args = args_parser()
    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f'{key}: {value}')

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    print(device)
    # load dataset and user groups
    train_loader, test_loader, global_test_loader, count_list = get_dataset(args)
    seed = 520
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # construct model
    if args.backbone == 'CNN':
        if args.dataset in ['cifar', 'cifar10', 'cinic', 'cinic_sep']:
            global_model = CifarCNN(num_classes=args.num_classes).to(device)
            args.lr = 0.02
        elif args.dataset == 'cifar100':
            args.num_classes = 100
            global_model = CifarCNN(num_classes=args.num_classes).to(device)
            args.lr = 0.02
        elif args.dataset == 'fmnist':
            global_model = CNN_FMNIST().to(device)
        elif args.dataset == 'emnist':
            args.num_classes = 62
            global_model = CNN_FMNIST(num_classes=args.num_classes).to(device)
        else:
            raise NotImplementedError()
    elif args.backbone == 'ResNet18':
        if args.dataset in ['cifar', 'cifar10', 'cinic', 'cinic_sep']:
            global_model = ResNet18(num_classes=args.num_classes).to(device)
            args.lr = 0.02
        elif args.dataset == 'cifar100':
            args.num_classes = 100
            global_model = CifarCNN(num_classes=args.num_classes).to(device)
            args.lr = 0.02
        elif args.dataset == 'fmnist':
            global_model = CNN_FMNIST().to(device)
        elif args.dataset == 'emnist':
            args.num_classes = 62
            global_model = CNN_FMNIST(num_classes=args.num_classes).to(device)
        else:
            raise NotImplementedError()

    # Training Rule
    LocalUpdate = local_update(args.train_rule)
    # One Round Training Function
    train_round_parallel = one_round_training(args.train_rule)

    # Training
    train_loss, train_acc = [], []
    test_acc = []
    local_accs1, local_accs2 = [], []
    # ======================================================================================================#
    local_clients = []
    for idx in range(args.num_users):
        local_clients.append(LocalUpdate(idx=idx, args=args, train_set=train_loader[idx], test_set=test_loader[idx],
                                         model=copy.deepcopy(global_model)))

    # add start
    max_acc1 = 0
    max_acc2 = 0
    average_value = []
    # add end

    for round in range(args.epochs):
        loss1, loss2, local_acc1, local_acc2, max_acc1, max_acc2, average_value = train_round_parallel(args, global_model,
                                                                                        local_clients, round, max_acc1,
                                                                                        max_acc2, count_list, average_value)
        train_loss.append(loss1)
        print("Train Loss: {}, {}".format(loss1, loss2))
        print("Local Accuracy on Local Data: {}%, {}%".format(local_acc1, local_acc2))
        local_accs1.append(local_acc1)
        local_accs2.append(local_acc2)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"time cost: {elapsed_time}s")


