# -*- coding: UTF-8 -*-
import numpy as np
import scipy.io as io
import torch
import MyDataset
import MyLoss
import model
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import utils
from datetime import datetime
import os
import time
from utils import Logger, time_to_str
from timeit import default_timer as timer
import time
import random
from tqdm import tqdm

TARGET_DOMAIN = {'VIPL': ['V4V', 'PURE', 'BUAA', 'UBFC'], \
                 'V4V': ['VIPL', 'PURE', 'BUAA', 'UBFC'], \
                 'PURE': ['VIPL', 'V4V', 'BUAA', 'UBFC'], \
                 'BUAA': ['VIPL', 'V4V', 'PURE', 'UBFC'], \
                 'UBFC': ['VIPL', 'V4V', 'PURE', 'BUAA']}

FILEA_NAME = {'VIPL': ['VIPL', 'VIPL', 'STMap_RGB_Align_CSI'], \
              'V4V': ['V4V', 'V4V', 'STMap_RGB'], \
              'PURE': ['PURE', 'PURE', 'STMap'], \
              'BUAA': ['BUAA', 'BUAA', 'STMap_RGB'], \
              'UBFC': ['UBFC', 'UBFC', 'STMap']}

if __name__ == '__main__':
    args = utils.get_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    Source_domain_Names = TARGET_DOMAIN[args.tgt]
    root_file = r'./STMap/'
    # 参数
    File_Name_0 = FILEA_NAME[Source_domain_Names[0]]
    source_name_0 = Source_domain_Names[0]
    source_fileRoot_0 = root_file + File_Name_0[0]
    source_saveRoot_0 = root_file + 'STMap_Index/' + File_Name_0[1]
    source_map_0 = File_Name_0[2] + '.png'

    File_Name_1 = FILEA_NAME[Source_domain_Names[1]]
    source_name_1 = Source_domain_Names[1]
    source_fileRoot_1 = root_file + File_Name_1[0]
    source_saveRoot_1 = root_file + 'STMap_Index/' + File_Name_1[1]
    source_map_1 = File_Name_1[2] + '.png'

    File_Name_2 = FILEA_NAME[Source_domain_Names[2]]
    source_name_2 = Source_domain_Names[2]
    source_fileRoot_2 = root_file + File_Name_2[0]
    source_saveRoot_2 = root_file + 'STMap_Index/' + File_Name_2[1]
    source_map_2 = File_Name_2[2] + '.png'

    File_Name_3 = FILEA_NAME[Source_domain_Names[3]]
    source_name_3 = Source_domain_Names[3]
    source_fileRoot_3 = root_file + File_Name_3[0]
    source_saveRoot_3 = root_file + 'STMap_Index/' + File_Name_3[1]
    source_map_3 = File_Name_3[2] + '.png'

    FILE_Name = FILEA_NAME[args.tgt]
    Target_name = args.tgt
    Target_fileRoot = root_file + FILE_Name[0]
    Target_saveRoot = root_file + 'STMap_Index/' + FILE_Name[1]
    Target_map = FILE_Name[2] + '.png'

    # 训练参数
    batch_size_num = args.batchsize
    epoch_num = args.epochs
    learning_rate = args.lr

    test_batch_size = args.batchsize
    num_workers = args.num_workers
    GPU = args.GPU

    # 图片参数
    input_form = args.form
    reTrain = args.reTrain
    frames_num = args.frames_num
    fold_num = args.fold_num
    fold_index = args.fold_index

    best_mae = 99

    print('batch num:', batch_size_num, ' epoch_num:', epoch_num, ' GPU Inedex:', GPU)
    print(' frames num:', frames_num, ' learning rate:', learning_rate, )
    print('fold num:', frames_num, ' fold index:', fold_index)

    if not os.path.exists('./Result_log'):
        os.makedirs('./Result_log')
    rPPGNet_name = 'rPPGNet_' + Target_name + 'Spatial' + str(args.spatial_aug_rate) + 'Temporal' + str(
        args.temporal_aug_rate)
    log = Logger()
    log.open('./Result_log/' + rPPGNet_name + '_log.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

    # 运行媒介
    if torch.cuda.is_available():
        device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu')  #
        print('on GPU')
    else:
        print('on CPU')

    # 数据集
    if args.reData == 1:
        source_index_0 = os.listdir(source_fileRoot_0)
        source_index_1 = os.listdir(source_fileRoot_1)
        source_index_2 = os.listdir(source_fileRoot_2)
        source_index_3 = os.listdir(source_fileRoot_3)
        Target_index = os.listdir(Target_fileRoot)

        source_Indexa_0 = MyDataset.getIndex(source_fileRoot_0, source_index_0, \
                                             source_saveRoot_0, source_map_0, 10, frames_num)
        source_Indexa_1 = MyDataset.getIndex(source_fileRoot_1, source_index_1, \
                                             source_saveRoot_1, source_map_1, 10, frames_num)
        source_Indexa_2 = MyDataset.getIndex(source_fileRoot_2, source_index_2, \
                                             source_saveRoot_2, source_map_2, 10, frames_num)
        source_Indexa_3 = MyDataset.getIndex(source_fileRoot_3, source_index_3, \
                                             source_saveRoot_3, source_map_3, 10, frames_num)
        Target_Indexa = MyDataset.getIndex(Target_fileRoot, Target_index, \
                                           Target_saveRoot, Target_map, 10, frames_num)

    source_db_0 = MyDataset.Data_DG(root_dir=source_saveRoot_0, dataName=source_name_0, \
                                    STMap=source_map_0, frames_num=frames_num, args=args, domain_label=0)
    source_db_1 = MyDataset.Data_DG(root_dir=source_saveRoot_1, dataName=source_name_1, \
                                    STMap=source_map_1, frames_num=frames_num, args=args, domain_label=1)
    source_db_2 = MyDataset.Data_DG(root_dir=source_saveRoot_2, dataName=source_name_2, \
                                    STMap=source_map_2, frames_num=frames_num, args=args, domain_label=2)
    source_db_3 = MyDataset.Data_DG(root_dir=source_saveRoot_3, dataName=source_name_3, \
                                    STMap=source_map_3, frames_num=frames_num, args=args, domain_label=3)
    Target_db = MyDataset.Data_DG(root_dir=Target_saveRoot, dataName=Target_name, \
                                  STMap=Target_map, frames_num=frames_num, args=args)

    src_loader_0 = DataLoader(source_db_0, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    src_loader_1 = DataLoader(source_db_1, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    src_loader_2 = DataLoader(source_db_2, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    src_loader_3 = DataLoader(source_db_3, batch_size=batch_size_num, shuffle=True, num_workers=num_workers)
    tgt_loader = DataLoader(Target_db, batch_size=batch_size_num, shuffle=False, num_workers=num_workers)

    my_model = model.HSRD()

    if reTrain == 1:
        my_model = torch.load('./Result_Model/xxx', # your pretrained model
                              map_location=device)
        print('load ' + rPPGNet_name + ' right')
    my_model.to(device=device)
    optimizer_rPPG = torch.optim.Adam(my_model.parameters(), lr=learning_rate)
    loss_func_NP = MyLoss.P_loss3().to(device)
    loss_func_L1 = nn.L1Loss().to(device)
    loss_contrastive = MyLoss.ContrastLoss().to(device)
    loss_adv = nn.CrossEntropyLoss().to(device)
    loss_func_SP = MyLoss.SP_loss(device, clip_length=frames_num).to(device)

    src_iter_0 = src_loader_0.__iter__()
    src_iter_per_epoch_0 = len(src_iter_0)

    src_iter_1 = src_loader_1.__iter__()
    src_iter_per_epoch_1 = len(src_iter_1)

    src_iter_2 = src_loader_2.__iter__()
    src_iter_per_epoch_2 = len(src_iter_2)

    src_iter_3 = src_loader_3.__iter__()
    src_iter_per_epoch_3 = len(src_iter_3)

    tgt_iter = iter(tgt_loader)
    tgt_iter_per_epoch = len(tgt_iter)

    max_iter = args.max_iter
    start = timer()
    with tqdm(range(max_iter + 1)) as it:
        for iter_num in it:
            my_model.train()
            if (iter_num % src_iter_per_epoch_0 == 0):
                src_iter_0 = src_loader_0.__iter__()
            if (iter_num % src_iter_per_epoch_1 == 0):
                src_iter_1 = src_loader_1.__iter__()
            if (iter_num % src_iter_per_epoch_2 == 0):
                src_iter_2 = src_loader_2.__iter__()
            if (iter_num % src_iter_per_epoch_3 == 0):
                src_iter_3 = src_loader_3.__iter__()

            ######### data prepare #########
            data0, bvp0, HR_rel0, data_aug0, bvp_aug0, HR_rel_aug0, domain_label0 = src_iter_0.__next__()
            data1, bvp1, HR_rel1, data_aug1, bvp_aug1, HR_rel_aug1, domain_label1 = src_iter_1.__next__()
            data2, bvp2, HR_rel2, data_aug2, bvp_aug2, HR_rel_aug2, domain_label2 = src_iter_2.__next__()
            data3, bvp3, HR_rel3, data_aug3, bvp_aug3, HR_rel_aug3, domain_label3 = src_iter_3.__next__()

            data0 = Variable(data0).float().to(device=device)
            bvp0 = Variable(bvp0).float().to(device=device).unsqueeze(dim=1)
            HR_rel0 = Variable(torch.Tensor(HR_rel0)).float().to(device=device)
            data_aug0 = Variable(data_aug0).float().to(device=device)
            bvp_aug0 = Variable(bvp_aug0).float().to(device=device).unsqueeze(dim=1)
            HR_rel_aug0 = Variable(torch.Tensor(HR_rel_aug0)).float().to(device=device)
            domain_label0 = domain_label0.long().to(device)

            data1 = Variable(data1).float().to(device=device)
            bvp1 = Variable((bvp1)).float().to(device=device).unsqueeze(dim=1)
            HR_rel1 = Variable(torch.Tensor(HR_rel1)).float().to(device=device)
            data_aug1 = Variable(data_aug1).float().to(device=device)
            bvp_aug1 = Variable((bvp_aug1)).float().to(device=device).unsqueeze(dim=1)
            HR_rel_aug1 = Variable(torch.Tensor(HR_rel_aug1)).float().to(device=device)
            domain_label1 = domain_label1.long().to(device)

            data2 = Variable(data2).float().to(device=device)
            bvp2 = Variable((bvp2)).float().to(device=device).unsqueeze(dim=1)
            HR_rel2 = Variable(torch.Tensor(HR_rel2)).float().to(device=device)
            data_aug2 = Variable(data_aug2).float().to(device=device)
            bvp_aug2 = Variable((bvp_aug2)).float().to(device=device).unsqueeze(dim=1)
            HR_rel_aug2 = Variable(torch.Tensor(HR_rel_aug2)).float().to(device=device)
            domain_label2 = domain_label2.long().to(device)

            data3 = Variable(data3).float().to(device=device)
            bvp3 = Variable((bvp3)).float().to(device=device).unsqueeze(dim=1)
            HR_rel3 = Variable(torch.Tensor(HR_rel3)).float().to(device=device)
            data_aug3 = Variable(data_aug3).float().to(device=device)
            bvp_aug3 = Variable((bvp_aug3)).float().to(device=device).unsqueeze(dim=1)
            HR_rel_aug3 = Variable(torch.Tensor(HR_rel_aug3)).float().to(device=device)
            domain_label3 = domain_label3.long().to(device)

            optimizer_rPPG.zero_grad()
            d_b0, d_b1, d_b2, d_b3 = data0.shape[0], data1.shape[0], data2.shape[0], data3.shape[0]

            input = torch.cat([data0, data1, data2, data3], dim=0)
            input_aug = torch.cat([data_aug0, data_aug1, data_aug2, data_aug3], dim=0)
            HR_rel = torch.cat([HR_rel0, HR_rel1, HR_rel2, HR_rel3], dim=0)


            rand_idx = torch.randperm(input.shape[0])
            HR_pr, bvp_pre, fea_x1_x1, fea_x1_x2, domain_invariant = my_model(input, input[rand_idx, :, :, :])
            HR_pr_aug, bvp_pre_aug, fea_x1_x1_aug, fea_x1_x2_aug, domain_invariant_aug = my_model(input_aug,
                                                                                                  input_aug[rand_idx, :,
                                                                                                  :,
                                                                                                  :])
      

            bvp_pre0, bvp_pre1, bvp_pre2, bvp_pre3 = bvp_pre[0:d_b0], bvp_pre[d_b0:d_b0 + d_b1], bvp_pre[
                                                                                                 d_b0 + d_b1:d_b0 + d_b1 + d_b2], bvp_pre[
                                                                                                                                  d_b0 + d_b1 + d_b2:]
            HR_pr0, HR_pr1, HR_pr2, HR_pr3 = HR_pr[0:d_b0], HR_pr[d_b0:d_b0 + d_b1], HR_pr[
                                                                                     d_b0 + d_b1:d_b0 + d_b1 + d_b2], HR_pr[
                                                                                                                      d_b0 + d_b1 + d_b2:]
            bvp_pre_aug0, bvp_pre_aug1, bvp_pre_aug2, bvp_pre_aug3 = bvp_pre_aug[0:d_b0], bvp_pre_aug[
                                                                                          d_b0:d_b0 + d_b1], bvp_pre_aug[
                                                                                                             d_b0 + d_b1:d_b0 + d_b1 + d_b2], bvp_pre_aug[
                                                                                                                                              d_b0 + d_b1 + d_b2:]
            HR_pr_aug0, HR_pr_aug1, HR_pr_aug2, HR_pr_aug3 = HR_pr_aug[0:d_b0], HR_pr_aug[d_b0:d_b0 + d_b1], HR_pr_aug[
                                                                                                             d_b0 + d_b1:d_b0 + d_b1 + d_b2], HR_pr_aug[
                                                                                                                                              d_b0 + d_b1 + d_b2:]

            src_loss_0 = MyLoss.get_loss(bvp_pre0, HR_pr0, bvp0, HR_rel0, source_name_0, \
                                         loss_func_NP, loss_func_L1, args, iter_num)
            src_loss_1 = MyLoss.get_loss(bvp_pre1, HR_pr1, bvp1, HR_rel1, source_name_1, \
                                         loss_func_NP, loss_func_L1, args, iter_num)
            src_loss_2 = MyLoss.get_loss(bvp_pre2, HR_pr2, bvp2, HR_rel2, source_name_2, \
                                         loss_func_NP, loss_func_L1, args, iter_num)
            src_loss_3 = MyLoss.get_loss(bvp_pre3, HR_pr3, bvp3, HR_rel3, source_name_3, \
                                         loss_func_NP, loss_func_L1, args, iter_num)

            src_loss_aug_0 = MyLoss.get_loss(bvp_pre_aug0, HR_pr_aug0, bvp_aug0, HR_rel_aug0, source_name_0, \
                                             loss_func_NP, loss_func_L1, args, iter_num)
            src_loss_aug_1 = MyLoss.get_loss(bvp_pre_aug1, HR_pr_aug1, bvp_aug1, HR_rel_aug1, source_name_1, \
                                             loss_func_NP, loss_func_L1, args, iter_num)
            src_loss_aug_2 = MyLoss.get_loss(bvp_pre_aug2, HR_pr_aug2, bvp_aug2, HR_rel_aug2, source_name_2, \
                                             loss_func_NP, loss_func_L1, args, iter_num)
            src_loss_aug_3 = MyLoss.get_loss(bvp_pre_aug3, HR_pr_aug3, bvp_aug3, HR_rel_aug3, source_name_3, \
                                             loss_func_NP, loss_func_L1, args, iter_num)

            HR_rels = torch.cat((HR_rel0, HR_rel1, HR_rel2, HR_rel3), dim=0)
            HR_rel_augs = torch.cat((HR_rel_aug0, HR_rel_aug1, HR_rel_aug2, HR_rel_aug3), dim=0)

            k = 2.0 / (1.0 + np.exp(-10.0 * iter_num / args.max_iter)) - 1.0

            loss = (src_loss_0 + src_loss_1 + src_loss_2 + src_loss_3) \
                   + (src_loss_aug_0 + src_loss_aug_1 + src_loss_aug_2 + src_loss_aug_3)

            domain_label = torch.cat([domain_label0, domain_label1, domain_label2, domain_label3], dim=0)

            k = 2.0 / (1.0 + np.exp(-10.0 * iter_num / args.max_iter)) - 1.0
            l_a = (0.001 * k) * (
                    loss_adv(domain_invariant, domain_label) + loss_adv(domain_invariant_aug, domain_label))
            loss += l_a
            n = HR_rels.shape[0]

            tag4contrast = HR_rels.unsqueeze(0).repeat(n, 1)
            topk = int(args.tk) if int(args.tk) <= n else n
            tag4contrast_idx = torch.topk(
                -1 * abs(tag4contrast - tag4contrast.permute(1, 0)) / tag4contrast.permute(1, 0),
                k=topk, dim=0).indices
            tag4contrast = torch.full((n, n), -1)
            tag4contrast_idx = tag4contrast_idx.permute(1, 0)
            for i in range(n):
                for j in tag4contrast_idx[i]:
                    tag4contrast[i, j] = 1
            tag4contrast = tag4contrast.to(device)

            con_loss = 0
            con_loss_aug = 0

            for i in range(n):
                con_loss += (0.01 * k) * loss_contrastive(fea_x1_x1[i], fea_x1_x2, tag4contrast[i])
                con_loss_aug += loss_contrastive(fea_x1_x1_aug[i], fea_x1_x2_aug, tag4contrast[i])
            con_loss = (0.01 * k) * ((con_loss / n) + (con_loss_aug / n))
            loss += con_loss

            if torch.sum(torch.isnan(loss)) > 0:
                print('Nan')
                break
            else:
                loss.backward()
                optimizer_rPPG.step()
                it.set_postfix(
                    ordered_dict={
                        "Train Inter": iter_num,
                        "loss": loss.data.cpu().numpy()
                    },
                    refresh=False,
                )
     
            if iter_num % 10 == 0:
                log.write(
                    'Train Inter:' + str(iter_num) \
                    + ' | loss:  ' + str(loss.data.cpu().numpy()) \
                    + ' |' + source_name_0 + ' : ' + str(src_loss_0.data.cpu().numpy()) \
                    + ' |' + source_name_1 + ' : ' + str(src_loss_1.data.cpu().numpy()) \
                    + ' |' + source_name_2 + ' : ' + str(src_loss_2.data.cpu().numpy()) \
                    + ' |' + source_name_3 + ' : ' + str(src_loss_3.data.cpu().numpy()) \
                    + ' |adversarial loss : ' + str(l_a.data.cpu().numpy()) \
                    + ' |contrastive loss : ' + str(con_loss.data.cpu().numpy()) \
                    + ' |' + time_to_str(timer() - start, 'min'))
                log.write('\n')

            if (iter_num >= 0) and (iter_num % 400 == 0):
                # Test
                print('Test:\n')
                my_model.eval()
                loss_mean = []
                Label_pr = []
                Label_gt = []
                HR_pr_temp = []
                HR_rel_temp = []
                HR_pr2_temp = []
                BVP_ALL = []
                BVP_PR_ALL = []
                for step, (data, bvp, HR_rel, _, _, _) in tqdm(enumerate(tgt_loader)):
                    data = Variable(data).float().to(device=device)
                    bvp = Variable(bvp).float().to(device=device)
                    HR_rel = Variable(HR_rel).float().to(device=device)
                    bvp = bvp.unsqueeze(dim=1)
                    Wave = bvp
                    rand_idx = torch.randperm(data.shape[0])
                    HR_pr, Wave_pr, fea_x1_x1, fea_x1_x2, domain_invariant = my_model(data, data[rand_idx, :, :, :])


                    if Target_name in ['VIPL', 'V4V', 'PURE']:
                        HR_rel_temp.extend(HR_rel.data.cpu().numpy())
                        HR_pr_temp.extend(HR_pr.data.cpu().numpy())
                        BVP_ALL.extend(Wave.data.cpu().numpy())
                        BVP_PR_ALL.extend(Wave_pr.data.cpu().numpy())
                    else:
                        temp, HR_rel = loss_func_SP(Wave, HR_rel)
                        HR_rel_temp.extend(HR_rel.data.cpu().numpy())
                        temp, HR_pr = loss_func_SP(Wave_pr, HR_pr)
                        HR_pr_temp.extend(HR_pr.data.cpu().numpy())
                        BVP_ALL.extend(Wave.data.cpu().numpy())
                        BVP_PR_ALL.extend(Wave_pr.data.cpu().numpy())

                print('HR:')
                ME, STD, MAE, RMSE, MER, P = utils.MyEval(HR_pr_temp, HR_rel_temp)
                log.write(
                    'Test Inter:' + str(iter_num) \
                    + ' | ME:  ' + str(ME) \
                    + ' | STD: ' + str(STD) \
                    + ' | MAE: ' + str(MAE) \
                    + ' | RMSE: ' + str(RMSE) \
                    + ' | MER: ' + str(MER) \
                    + ' | P ' + str(P))
                log.write('\n')

                if not os.path.exists('./Result_Model'):
                    os.makedirs('./Result_Model')

                # if best_mae > MAE:
                # best_mae = MAE
                if not os.path.exists('./Result'):
                    os.makedirs('./Result')
                io.savemat('./Result/' + rPPGNet_name + '_' + str(
                    args.tp) + '_' + str(args.tk)+ 'without both' + '_HR_pr.mat', {'HR_pr': HR_pr_temp})
                io.savemat('./Result/' + rPPGNet_name + '_' + str(
                    args.tp) + '_' + str(args.tk)+ 'without both' + '_HR_rel.mat', {'HR_rel': HR_rel_temp})
                io.savemat('./Result/' + rPPGNet_name + '_' + str(
                    args.tp) + '_' + str(args.tk)+ 'without both' + '_WAVE_ALL.mat',
                           {'Wave': BVP_ALL})
                io.savemat('./Result/' + rPPGNet_name + '_' + str(
                    args.tp) + '_' + str(args.tk)+ 'without both' + '_WAVE_PR_ALL.mat',
                           {'Wave': BVP_PR_ALL})
                torch.save(my_model,
                           './Result_Model/' + rPPGNet_name + '_' + str(
                               args.tp) + '_' + str(args.tk) + 'without both')
                print('saveModel As ' + rPPGNet_name)
