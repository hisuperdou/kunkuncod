# import torch
# import torch.backends.cudnn as cudnn
# import torch.nn as nn
# from torch import optim
# from torch.autograd import Variable
# import torch.multiprocessing as mp
# import torch.distributed as dist

# from dataset import get_loader
# import math
# from Models.ImageDepthNet import ImageDepthNet
# import os

# # os.environ["CUDA_VISIBLE_DEVICES"] = "0 ，1"
# # device_ids = [0, 1]

# # 自己加的
# # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'#设置所有可以使用的显卡，共计2块
# # local_rank = [0,1]

# def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
#     fh = open(save_dir, 'a')
#     epoch_total_loss = str(epoch_total_loss)
#     epoch_loss = str(epoch_loss)
#     fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
#     fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
#     fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
#     fh.write('\n')
#     fh.close()


# def adjust_learning_rate(optimizer, decay_rate=.1):
#     update_lr_group = optimizer.param_groups
#     for param_group in update_lr_group:
#         print('before lr: ', param_group['lr'])
#         param_group['lr'] = param_group['lr'] * decay_rate
#         print('after lr: ', param_group['lr'])
#     return optimizer


# def save_lr(save_dir, optimizer):
#     update_lr_group = optimizer.param_groups[0]
#     fh = open(save_dir, 'a')
#     fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
#     fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
#     fh.write('\n')
#     fh.close()


# def train_net(num_gpus, args):
#     print(6962612)
#     print(num_gpus)
#     print(6962612)

#     mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))


# def main(local_rank, num_gpus, args):

#     cudnn.benchmark = True
#     dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=num_gpus, rank=local_rank)
#     #到这里直接重新进入了
#     torch.cuda.set_device(local_rank)

#     # os.environ["CUDA_VISIBLE_DEVICES"] = "0 ,1"
    
#     net = ImageDepthNet(args)
    
#     # net=nn.DataParallel(net,device_ids=[1,0])
#     # net=nn.DataParallel(net)



#     net.train()
#     net.cuda()

#     net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
#     net = torch.nn.parallel.DistributedDataParallel(
#         net,
#         device_ids=[local_rank],
#         output_device=local_rank,
#         find_unused_parameters=True)

#     base_params = [params for name, params in net.named_parameters() if ("backbone" in name)]
#     other_params = [params for name, params in net.named_parameters() if ("backbone" not in name)]

#     optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
#                             {'params': other_params, 'lr': args.lr}])
#     train_dataset = get_loader(args.trainset, args.data_root, args.img_size, mode='train')

#     sampler = torch.utils.data.distributed.DistributedSampler(
#         train_dataset,
#         num_replicas=num_gpus,
#         rank=local_rank,
#     )
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=6,
#                                                pin_memory=True,
#                                                sampler=sampler,
#                                                drop_last=True,
#                                                )

#     print('''
#         Starting training:
#             Train steps: {}
#             Batch size: {}
#             Learning rate: {}
#             Training size: {}
#         '''.format(args.train_steps, args.batch_size, args.lr, len(train_loader.dataset)))
#     print(777777777777777777777777)

#     N_train = len(train_loader) * args.batch_size

#     loss_weights = [1, 0.8, 0.8, 0.5, 0.5, 0.5]
#     if not os.path.exists(args.save_model_dir):
#         os.makedirs(args.save_model_dir)

#     criterion = nn.BCEWithLogitsLoss()
#     whole_iter_num = 0
#     iter_num = math.ceil(len(train_loader.dataset) / args.batch_size)
#     for epoch in range(args.epochs):

#         print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
#         print('epoch:{0}-------lr:{1}'.format(epoch + 1, args.lr))

#         epoch_total_loss = 0
#         epoch_loss = 0

#         for i, data_batch in enumerate(train_loader):
            
#             if (i + 1) > iter_num: break

#             images, depths, label_224, label_14, label_28, label_56, label_112, \
#             contour_224, contour_14, contour_28, contour_56, contour_112 = data_batch

#             images, depths, label_224, contour_224 = Variable(images.cuda(local_rank, non_blocking=True)), \
#                                         Variable(depths.cuda(local_rank, non_blocking=True)), \
#                                         Variable(label_224.cuda(local_rank, non_blocking=True)),  \
#                                         Variable(contour_224.cuda(local_rank, non_blocking=True))

#             label_14, label_28, label_56, label_112 = Variable(label_14.cuda()), Variable(label_28.cuda()),\
#                                                       Variable(label_56.cuda()), Variable(label_112.cuda())

#             contour_14, contour_28, contour_56, contour_112 = Variable(contour_14.cuda()), \
#                                                                                       Variable(contour_28.cuda()), \
#                                                       Variable(contour_56.cuda()), Variable(contour_112.cuda())

#             outputs_saliency, outputs_contour = net(images, depths)

#             mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
#             cont_1_16, cont_1_8, cont_1_4, cont_1_1 = outputs_contour
#             # loss
#             loss5 = criterion(mask_1_16, label_14)
#             loss4 = criterion(mask_1_8, label_28)
#             loss3 = criterion(mask_1_4, label_56)
#             loss1 = criterion(mask_1_1, label_224)

#             # contour loss
#             c_loss5 = criterion(cont_1_16, contour_14)
#             c_loss4 = criterion(cont_1_8, contour_28)
#             c_loss3 = criterion(cont_1_4, contour_56)
#             c_loss1 = criterion(cont_1_1, contour_224)

#             img_total_loss = loss_weights[0] * loss1 + loss_weights[2] * loss3 + loss_weights[3] * loss4 + loss_weights[4] * loss5
#             contour_total_loss = loss_weights[0] * c_loss1 + loss_weights[2] * c_loss3 + loss_weights[3] * c_loss4 + loss_weights[4] * c_loss5

#             total_loss = img_total_loss + contour_total_loss

#             epoch_total_loss += total_loss.cpu().data.item()
#             epoch_loss += loss1.cpu().data.item()
#             # 因为log暂时隐藏
#             # print(
#             #     'whole_iter_num: {0} --- {1:.4f} --- total_loss: {2:.6f} --- saliency loss: {3:.6f}'.format(
#             #         (whole_iter_num + 1),
#             #         (i + 1) * args.batch_size / N_train, total_loss.item(), loss1.item()))

#             optimizer.zero_grad()

#             total_loss.backward()

#             optimizer.step()
#             whole_iter_num += 1

#             if (local_rank == 0) and (whole_iter_num == args.train_steps):
#                 torch.save(net.state_dict(),
#                            args.save_model_dir + 'RGBD_VST.pth')

#             if whole_iter_num == args.train_steps:
#                 return 0

#             if whole_iter_num == args.stepvalue1 or whole_iter_num == args.stepvalue2 or whole_iter_num == args.stepvalue3:
                
#                 optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)
#                 save_dir = './loss.txt'
#                 save_lr(save_dir, optimizer)
#                 print('have updated lr!!')

#         print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))
#         save_lossdir = './loss.txt'
#         save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss/iter_num, epoch+1)




import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.distributed as dist

from dataset import get_loader
import math
from Models.ImageDepthNet import ImageDepthNet
import os
#后加第一支路的库
import torch.nn.functional as F
# from utils.utils import clip_gradient



# 自己修改代码部分：自定义loss计算，监督模型中间image_Input_tem_cnn的结果
def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()









#----------------------------------------------------------------------------------------
#传入的参数
# save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss/iter_num, epoch+1,epoch_loss_cnn/iter_num)
#下面形参是cnn_epoch后来加入
def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch, cnn_epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    #自己定义,不用暂时注释掉
    # fh.write(str(epoch) + '_cnn_epoch_loss' + str(cnn_epoch) + '\n')
    

    # 效果
    # until_99_run_iter_num148500
    # 99_epoch_total_loss0.07867544755747591
    # 99_epoch_loss0.018306136107127866
    
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer


def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()


def train_net(num_gpus, args):


    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))


def main(local_rank, num_gpus, args):

    cudnn.benchmark = True

    dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=num_gpus, rank=local_rank)

    torch.cuda.set_device(local_rank)

    net = ImageDepthNet(args)
    net.train()
    net.cuda()

    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True)

    base_params = [params for name, params in net.named_parameters() if ("backbone" in name)]
    other_params = [params for name, params in net.named_parameters() if ("backbone" not in name)]

    optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
                            {'params': other_params, 'lr': args.lr}])
    train_dataset = get_loader(args.trainset, args.data_root, args.img_size, mode='train')

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=6,
                                               pin_memory=True,
                                               sampler=sampler,
                                               drop_last=True,
                                               )

    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
        '''.format(args.train_steps, args.batch_size, args.lr, len(train_loader.dataset)))

    N_train = len(train_loader) * args.batch_size

    loss_weights = [1, 0.8, 0.8, 0.5, 0.5, 0.5]
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    criterion = nn.BCEWithLogitsLoss()
    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / args.batch_size)


    for epoch in range(args.epochs):
        
	    
        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, args.lr))
      



        epoch_total_loss = 0
        epoch_loss = 0
        epoch_loss_cnn = 0

        for i, data_batch in enumerate(train_loader):
            if (i + 1) > iter_num: break

            images, depths, label_224, label_14, label_28, label_56, label_112, \
            contour_224, contour_14, contour_28, contour_56, contour_112 = data_batch

            images, depths, label_224, contour_224 = Variable(images.cuda(local_rank, non_blocking=True)), \
                                        Variable(depths.cuda(local_rank, non_blocking=True)), \
                                        Variable(label_224.cuda(local_rank, non_blocking=True)),  \
                                        Variable(contour_224.cuda(local_rank, non_blocking=True))


            label_14, label_28, label_56, label_112 = Variable(label_14.cuda()), Variable(label_28.cuda()),\
                                                      Variable(label_56.cuda()), Variable(label_112.cuda())

            contour_14, contour_28, contour_56, contour_112 = Variable(contour_14.cuda()), \
                                                                                      Variable(contour_28.cuda()), \
                                                      Variable(contour_56.cuda()), Variable(contour_112.cuda())
            preds = net(images, depths)

            #第一支路的loss--------------------------------------------------------------------------
            # net(images, depths)[4]
            # loss_init = structure_loss(preds[0], label_224) + structure_loss(preds[1], label_224) + structure_loss(preds[2], label_224)
            # loss_final = structure_loss(preds[3], label_224)
            # loss = loss_init + loss_final
    
            # epoch_loss_cnn += loss.cpu().data.item()
            # optimizer.zero_grad()
            # # loss.backward()

            # clip_gradient(optimizer, opt.clip)
            # optimizer.step()

            #第一支路结尾----------------------------------------------------------------------------


            outputs_saliency, outputs_contour = preds

            mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
            cont_1_16, cont_1_8, cont_1_4, cont_1_1 = outputs_contour
            # loss
            loss5 = criterion(mask_1_16, label_14)
            loss4 = criterion(mask_1_8, label_28)
            loss3 = criterion(mask_1_4, label_56)
            loss1 = criterion(mask_1_1, label_224)
            # print(label_224.shape)
            # torch.Size([B, 1, 224, 224])     b取决于batch_size大小。
            # contour loss
            c_loss5 = criterion(cont_1_16, contour_14)
            c_loss4 = criterion(cont_1_8, contour_28)
            c_loss3 = criterion(cont_1_4, contour_56)
            c_loss1 = criterion(cont_1_1, contour_224)

            img_total_loss = loss_weights[0] * loss1 + loss_weights[2] * loss3 + loss_weights[3] * loss4 + loss_weights[4] * loss5
            contour_total_loss = loss_weights[0] * c_loss1 + loss_weights[2] * c_loss3 + loss_weights[3] * c_loss4 + loss_weights[4] * c_loss5

            # total_loss = img_total_loss + contour_total_loss  原来的，下面是进阶修改后
            # total_loss = img_total_loss + contour_total_loss + loss
            total_loss = img_total_loss + contour_total_loss
            

            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss += loss1.cpu().data.item()
            

            #输出，原来的。暂时注释
            print(
                'whole_iter_num: {0} --- {1:.4f} --- total_loss: {2:.6f} --- cod loss: {3:.6f}'.format(
                    (whole_iter_num + 1),
                    (i + 1) * args.batch_size / N_train, total_loss.item(), loss1.item()))
            
            optimizer.zero_grad()

            total_loss.backward()
            optimizer.step()
            #第一支路的输出部分，主要是增加了第一支路的loss
            # print('whole_iter_num: {0} --- {1:.4f} --- total_loss: {2:.6f} --- T_cod loss: {3:.6f} --- cnn_cod loss: {4:.6f}'.format(
            #         (whole_iter_num + 1),
            #         (i + 1) * args.batch_size / N_train, total_loss.item(), loss1.item(), loss.item()))
            #上面结束，下面是另外阶段
            whole_iter_num += 1

            if (local_rank == 0) and (whole_iter_num == args.train_steps):
                torch.save(net.state_dict(),
                           args.save_model_dir + 'RGBD_VST.pth')

            if whole_iter_num == args.train_steps:
                return 0

            if whole_iter_num == args.stepvalue1 or whole_iter_num == args.stepvalue2 or whole_iter_num == args.stepvalue3:
                optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)
                save_dir = './loss.txt'
                save_lr(save_dir, optimizer)
                print('have updated lr!!')



            # if whole_iter_num > 10000 or whole_iter_num % 10000 == 0:
            #     optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)
            #     save_dir = './loss.txt'
            #     save_lr(save_dir, optimizer)
            #     print('have updated lr!!')





        print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))
        save_lossdir = './loss.txt'
        # save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss/iter_num, epoch+1, epoch_loss_cnn/iter_num)
        save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss/iter_num, epoch+1, 0) 

        







