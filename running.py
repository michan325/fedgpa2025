import torch
from torch import nn
import tools
import numpy as np
import copy
import math
import json
from tools import average_weights_weighted, get_head_agg_weight, agg_classifier_weighted_p, \
    average_weights_weighted_ours
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon


def one_round_training(rule):
    # gradient aggregation rule
    Train_Round = {'FedAvg': train_round_fedavg,
                   'LG_FedAvg': train_round_lgfedavg,
                   'FedPer': train_round_fedper,
                   'Local': train_round_standalone,
                   'FedPAC': train_round_fedpac,
                   'FedGPA': train_round_fedgpa,
                   }
    return Train_Round[rule]


# training methods -------------------------------------------------------------------
# local training only

def train_round_standalone(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    print(idx_users)
    local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []

    global_weight = global_model.state_dict()

    for idx in idx_users:
        local_client = local_clients[idx]
        local_epoch = args.local_epoch
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch)
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2


# vanila FedAvg
def train_round_fedavg(args, global_model, local_clients, rnd, max_acc1, max_acc2, count_list, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []

    global_weight = global_model.state_dict()

    for idx in idx_users:
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight)
        local_epoch = args.local_epoch
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch, round=rnd)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    global_weight = average_weights_weighted(local_weights, agg_weight)
    # update global model
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2


# parameter decoupling
def train_round_lgfedavg(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_grads = []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []

    global_weight = global_model.state_dict()

    for idx in idx_users:
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight)
        local_epoch = args.local_epoch
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    global_weight = average_weights_weighted(local_weights, agg_weight)
    # global_weight = average_weights(local_weights)
    # update global model
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2


def train_round_fedper(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_grads = []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []

    global_weight = global_model.state_dict()

    for idx in idx_users:
        local_client = local_clients[idx]
        local_epoch = args.local_epoch
        agg_weight.append(local_client.agg_weight)
        local_client.update_local_model(global_weight=global_weight)
        w, loss1, loss2, acc1, acc2 = local_client.local_training(local_epoch=local_epoch, round=rnd)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(copy.deepcopy(loss1))
        local_losses2.append(copy.deepcopy(loss2))
        local_acc1.append(acc1)
        local_acc2.append(acc2)

    # get global weights
    global_weight = average_weights_weighted(local_weights, agg_weight)
    # update global model
    global_model.load_state_dict(global_weight)

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2


def train_round_fedpac(args, global_model, local_clients, rnd, max_acc1, max_acc2, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    print(idx_users)

    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []  # aggregation weights for f
    avg_weight = []  # aggregation weights for g
    sizes_label = []
    local_protos = []

    Vars = []
    Hs = []

    agg_g = args.agg_g  # conduct classifier aggregation or not

    if rnd <= args.epochs:
        for idx in idx_users:
            local_client = local_clients[idx]
            # statistics collection
            v, h = local_client.statistics_extraction()
            Vars.append(copy.deepcopy(v))
            Hs.append(copy.deepcopy(h))
            # local training
            local_epoch = args.local_epoch
            sizes_label.append(local_client.sizes_label)
            w, loss1, loss2, acc1, acc2, protos = local_client.local_training(local_epoch=local_epoch, round=rnd)
            local_weights.append(copy.deepcopy(w))
            local_losses1.append(copy.deepcopy(loss1))
            local_losses2.append(copy.deepcopy(loss2))
            local_acc1.append(round(acc1, 1))
            local_acc2.append(round(acc2, 1))
            agg_weight.append(local_client.agg_weight)
            local_protos.append(copy.deepcopy(protos))

        # get weight for feature extractor aggregation
        agg_weight = torch.stack(agg_weight).to(args.device)

        # update global feature extractor
        global_weight_new = average_weights_weighted(local_weights, agg_weight)

        # update global prototype
        global_protos = tools.protos_aggregation(local_protos, sizes_label)

        for idx in range(num_users):
            local_client = local_clients[idx]
            local_client.update_base_model(global_weight=global_weight_new)
            local_client.update_global_protos(global_protos=global_protos)

        # get weight for local classifier aggregation
        if agg_g and rnd < args.epochs:
            avg_weights = get_head_agg_weight(m, Vars, Hs)
            idxx = 0
            for idx in idx_users:
                local_client = local_clients[idx]
                if avg_weights[idxx] is not None:
                    new_cls = agg_classifier_weighted_p(local_weights, avg_weights[idxx], local_client.w_local_keys,
                                                        idxx)
                else:
                    new_cls = local_weights[idxx]
                local_client.update_local_classifier(new_weight=new_cls)
                idxx += 1

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    print(f'RESULT >>>>> local acc1 {local_acc1}\nRESULT >>>>> local acc2 {local_acc2}')

    if acc_avg1 > max_acc1:
        max_acc1 = acc_avg1

    if acc_avg2 > max_acc2:
        max_acc2 = acc_avg2

    print(f'MAX RESULT >>>>> max_acc1 {max_acc1}\nMAX RESULT >>>>> max_acc2 {max_acc2}')

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2, max_acc1, max_acc2

def train_round_fedgpa(args, global_model, local_clients, rnd, max_acc1, max_acc2, count_list, average_value, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    print(idx_users)
    local_weights, local_losses1, local_losses2 = [], [], []
    local_acc1 = []
    local_acc2 = []
    agg_weight = []  # aggregation weights for f
    avg_weight = []  # aggregation weights for g
    sizes_label = []
    local_protos = []

    Vars = []
    Hs = []
    # add start
    feature_matrix = []
    feature_dict_list = []
    # add end

    agg_g = args.agg_g  # conduct classifier aggregation or not

    if rnd <= args.epochs:
        for idx in idx_users:
            local_client = local_clients[idx]

            # statistics collection
            v, h, feature, feature_dict = local_client.statistics_extraction()
            Vars.append(copy.deepcopy(v))
            Hs.append(copy.deepcopy(h))
            feature_matrix.append(copy.deepcopy(feature))
            feature_dict_list.append(copy.deepcopy(feature_dict))

            # local training
            local_epoch = args.local_epoch
            sizes_label.append(local_client.sizes_label)
            w, loss1, loss2, acc1, acc2, protos = local_client.local_training(local_epoch=local_epoch, round=rnd,
                                                                              count_list=count_list[idx])


            local_weights.append(copy.deepcopy(w))
            local_losses1.append(copy.deepcopy(loss1))
            local_losses2.append(copy.deepcopy(loss2))
            local_acc1.append(round(acc1, 1))
            local_acc2.append(round(acc2, 1))
            agg_weight.append(local_client.agg_weight)
            local_protos.append(copy.deepcopy(protos))

        #
        agg_weight = torch.stack(agg_weight).to(args.device)
        agg_w = agg_weight / (agg_weight.sum(dim=0))

        # update global prototype
        global_protos = tools.protos_aggregation(local_protos, sizes_label)

        if args.plot == 1:
            client_num = 0
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown']
            markers = ['+', '^', 's', 'p', '*', 'D', 'v', '<', '>', 'x']
            tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=2023, init='random', n_iter=1000)

            v0 = feature_dict_list[client_num][0]
            v1 = feature_dict_list[client_num][1]
            v2 = feature_dict_list[client_num][2]
            v3 = feature_dict_list[client_num][3]
            v4 = feature_dict_list[client_num][4]
            v5 = feature_dict_list[client_num][5]
            v6 = feature_dict_list[client_num][6]
            v7 = feature_dict_list[client_num][7]
            v8 = feature_dict_list[client_num][8]
            v9 = feature_dict_list[client_num][9]

            v = np.concatenate(
                (v0.cpu(), v1.cpu(), v2.cpu(), v3.cpu(), v4.cpu(), v5.cpu(), v6.cpu(), v7.cpu(), v8.cpu(),
                 v9.cpu()), axis=0)

            vectors_2d = tsne.fit_transform(v)

            # create a blank plot
            plt.figure()

            start_index = 0
            class_num_list = []
            for i in range(len(feature_dict_list[client_num])):
                class_num_list.append(len(feature_dict_list[client_num][i]))
            # print(f'class num list: {class_num_list}')

            # record class centroid for each class
            class_centroid_list = []

            for i in range(len(feature_dict_list[client_num])):
                # plt.scatter(vectors_2d[start_index:start_index + class_num_list[i], 0],
                #             vectors_2d[start_index:start_index + class_num_list[i], 1],
                #             color=colors[i], label=f'class {i}')

                # zorder 用于控制中心散点始终在全部散点之上，相当于图层顺序
                # alpha 控制透明度
                plt.scatter(vectors_2d[start_index:start_index + class_num_list[i], 0],
                            vectors_2d[start_index:start_index + class_num_list[i], 1],
                            color=colors[i], alpha=0.6, zorder=1)

                # s: size
                plt.scatter(vectors_2d[start_index:start_index + class_num_list[i], 0].mean(),
                            vectors_2d[start_index:start_index + class_num_list[i], 1].mean(),
                            color=colors[i], marker='*', label=f'class {i}', s=200, zorder=2)

                class_centroid_list.append([vectors_2d[start_index:start_index + class_num_list[i], 0].mean(),
                                            vectors_2d[start_index:start_index + class_num_list[i], 1].mean()])
                start_index += class_num_list[i]

            plt.legend(loc='upper right', bbox_to_anchor=(1.27, 1), borderaxespad=0.5)
            # right expand
            plt.subplots_adjust(right=0.8)
            plt.title(f'\n-- Global Communication Round : {rnd + 1}')

            # folder_path = f'pictures/1125/{args.current_client}'
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)
            #
            # plt.savefig(folder_path + os.sep + f'{rnd + 1}.jpg', dpi=300)
            plt.show()


        weights_extractor = []
        weights_classifier = []


        if args.alpha_f != -1:
            for i in range(len(feature_matrix)):
                anchor = feature_matrix[i]

                if args.distance == 'euclidean':
                    euclidean_distances = [np.linalg.norm(anchor.cpu() - matrix.cpu(), axis=1) for matrix in feature_matrix]

                    sample_weight = count_list[idx_users[i]]
                    averaged_distance = []
                    for d in euclidean_distances:
                        averaged_distance.append(round(np.dot(sample_weight, d), 1))

                    averaged_distance = averaged_distance / count_list[idx_users[i]].sum()

                    min_nonzero = np.min(averaged_distance[np.nonzero(averaged_distance)])
                    averaged_distance[averaged_distance == 0] = min_nonzero / 2
                    # 计算反比例值
                    inverse_values = 1 / averaged_distance
                    # 归一化权重
                    weights = inverse_values / np.sum(inverse_values)
                    # prototype distance
                    averaged_distance = [round(x, 3) for x in weights]

                    # sample num
                    tmp_array = np.array(agg_w.cpu())

                    alpha = 0.1

                    combined_weights = alpha * tmp_array + (1 - alpha) * np.array(averaged_distance)
                    combined_weights /= np.sum(combined_weights)

                    averaged_distance = [round(x, 3) for x in combined_weights]


                if args.distance == 'cosine':
                    # cosine similarity
                    cosine = [cosine_similarity(anchor.cpu(), matrix.cpu()) for matrix in feature_matrix]
                    listA = []
                    for j in range(len(cosine)):
                        for k in range(cosine[j].shape[0]):
                            listA.append(cosine[j][k, k])

                    cosine_distances = []
                    for j in range(len(cosine)):
                        cosine_distances.append(np.array(listA[j*cosine[j].shape[0]: (j+1)*cosine[j].shape[0]]))

                    sample_weight = count_list[idx_users[i]]
                    averaged_distance = []
                    for d in cosine_distances:
                        averaged_distance.append(round(np.dot(sample_weight, d), 1))

                    averaged_distance = averaged_distance / count_list[idx_users[i]].sum()
                    # averaged_distance[i] = np.mean(averaged_distance)
                    averaged_distance = [round(x, 2) for x in averaged_distance]
                    if i == 0:
                        print(f'averaged_distance: {averaged_distance}')

                    filtered_list = [value for value in averaged_distance if value != 1]
                    negated_list = [args.alpha_f * x for x in filtered_list]
                    exp_list = np.exp(negated_list)
                    total_sum = sum(exp_list)
                    normalized_list = [x / total_sum for x in exp_list]

                    scaling_factor = (1 - agg_w[i].tolist()) / sum(normalized_list)
                    normalized_and_scaled_list = [x * scaling_factor for x in normalized_list]

                    normalized_and_scaled_list.insert(i, agg_w[i].tolist())
                    averaged_distance = [round(x, 3) for x in normalized_and_scaled_list]

                # weights_classifier.append(cls_temp)

                weights_extractor.append(averaged_distance)
                weights_classifier.append(averaged_distance)

            print(f'Weighted averaged distance: {averaged_distance}')
            if rnd % 50 == 0:
                print(f'Round {rnd}, All Weighted averaged distance: {weights_extractor}')
                with open("weights_extractor_output.txt", "a") as file:  # 使用 "a" 模式来追加写入，不覆盖之前内容
                    file.write(f"Round {rnd}:\n{str(weights_extractor)}\n\n")
            # print(f'Sample number weights: {sample_num_list}')

            # print(f'averaged_distance: {averaged_distance}')
            # print(f'Threshold distance for aggregating feature extractor: {average_value}')
        else:
            row_sums = count_list.sum(axis=1, keepdims=True)
            normalized_count_list = count_list / row_sums
            normalized_count_list = np.round(normalized_count_list, 3)
            length =normalized_count_list.shape[1]
            temp_array = np.full(length, 1/length)

            # 计算JS相似度
            js_similarities = []
            for row in normalized_count_list:
                js_similarity = jensenshannon(row, temp_array)
                js_similarities.append(js_similarity)

            js_similarities = np.round(js_similarities, 3)
            total_sum = sum(js_similarities)
            norm_js_similarities = [x / total_sum for x in js_similarities]
            norm_js_similarities = np.round(norm_js_similarities, 3)
            print(f'Feature Extractor Agg Weight: {norm_js_similarities.tolist()}')

        if args.alpha_f != -1:
            for idx in idx_users:
                local_client = local_clients[idx]
                # update global feature extractor
                if rnd >= args.epochs * args.ratio:
                    global_weight_new = average_weights_weighted_ours(local_weights, weights_extractor[idx_users.index(idx)])
                elif args.alpha_f == 0:
                    global_weight_new = average_weights_weighted_ours(local_weights, agg_w)
                else:
                    global_weight_new = average_weights_weighted_ours(local_weights, agg_w)

                local_client.update_base_model(global_weight=global_weight_new)
                local_client.update_global_protos(global_protos=global_protos)
        else:
            # global_weight_new = average_weights_weighted_ours(local_weights, norm_js_similarities)
            global_weight_new = average_weights_weighted_ours(local_weights, agg_w)
            global_protos = tools.protos_aggregation(local_protos, sizes_label)

            for idx in range(num_users):
                local_client = local_clients[idx]
                local_client.update_base_model(global_weight=global_weight_new)
                local_client.update_global_protos(global_protos=global_protos)

        # get weight for local classifier aggregation
        if agg_g and rnd < args.epochs:
            avg_weights = get_head_agg_weight(m, Vars, Hs, weights_extractor)
            if rnd >= args.epochs * args.ratio:

                idxx = 0
                for idx in idx_users:
                    local_client = local_clients[idx]
                    if avg_weights[idxx] is not None:
                        new_cls = agg_classifier_weighted_p(local_weights, avg_weights[idxx], local_client.w_local_keys,
                                                            idxx)
                    else:
                        new_cls = local_weights[idxx]
                    local_client.update_local_classifier(new_weight=new_cls)
                    idxx += 1

                # print(f"Classifier Weights: {weights_classifier[0]}")

            elif args.alpha_c == 0:
                # add start
                avg_weights = agg_w.tolist()
                # add end

                idxx = 0
                for idx in idx_users:
                    local_client = local_clients[idx]

                    new_cls = agg_classifier_weighted_p(local_weights, avg_weights, local_client.w_local_keys, idxx)

                    local_client.update_local_classifier(new_weight=new_cls)
                    idxx += 1

                print(f"Classifier Weights: {avg_weights}")

            else:
                # add start
                avg_weights = agg_w.tolist()
                # add end

                idxx = 0
                for idx in idx_users:
                    local_client = local_clients[idx]

                    new_cls = agg_classifier_weighted_p(local_weights, avg_weights, local_client.w_local_keys, idxx)

                    local_client.update_local_classifier(new_weight=new_cls)
                    idxx += 1

                print(f"Classifier Weights: {avg_weights}")

    loss_avg1 = sum(local_losses1) / len(local_losses1)
    loss_avg2 = sum(local_losses2) / len(local_losses2)
    acc_avg1 = sum(local_acc1) / len(local_acc1)
    acc_avg2 = sum(local_acc2) / len(local_acc2)

    print(f'RESULT >>>>> local acc1 {local_acc1}\nRESULT >>>>> local acc2 {local_acc2}')

    if acc_avg1 > max_acc1:
        max_acc1 = acc_avg1

    if acc_avg2 > max_acc2:
        max_acc2 = acc_avg2

    print(f'MAX RESULT >>>>> max_acc1 {max_acc1}\nMAX RESULT >>>>> max_acc2 {max_acc2}')

    return loss_avg1, loss_avg2, acc_avg1, acc_avg2, max_acc1, max_acc2, average_value
