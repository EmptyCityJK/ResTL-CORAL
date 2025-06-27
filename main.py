import os
import argparse
import numpy as np
import model
import utils
from torch.utils.data import DataLoader
import warnings
from train import *
from Dataset import OpenBMI_RS_MI_Dataset, OpenBMI_RSOnly_Dataset

warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 实验参数配置
parser = argparse.ArgumentParser(description="Cross-subject")
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-te", "--test", type=str, default='5', help='要作为测试集的被试编号')
parser.add_argument("-m", "--model", type=str, default='EEGNet')
parser.add_argument("-e", "--epoch", type=int, default=100)
parser.add_argument("-l", "--learningrate", type=float, default=0.0005)
parser.add_argument("-t", "--is_training", type=str, default='train')
parser.add_argument("-b", "--batch", type=int, default=128)
parser.add_argument("-d", "--data", type=str, default='OpenBMI', choices=['BCI4_2b', 'BCI4_2a', 'OpenBMI'])
parser.add_argument("--size", type=int, default=750) # 采样率统一为250Hz，3s*250 = 750点
parser.add_argument("--rest", type=bool, default=False)
parser.add_argument("--cdist", type=float, default = 1e-5)
parser.add_argument("--checkpoint", type=str, default='checkpoint')
parser.add_argument('--multiproto', type=int, default=1, help='Whether to use multiple prototypes (1 = no, >1 = yes)')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

torch.multiprocessing.set_sharing_strategy('file_system')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device
test = list(map(int, args.test.split(',')))

if args.data == 'BCI4_2b':
    test = list(map(int, args.test.split(',')))
    train = [idx for idx in list(range(1, 10)) if not idx in test]
    args.datanum = 2 # 分类数
    args.chnum = 3 # 通道数
    args.sessionnum = 5 # 会话数
if args.data == 'OpenBMI':
    test = list(map(int, args.test.split(',')))
    # train = [idx for idx in list(range(1, 55)) if not idx in test]
    # 除了 test 中指定的 subject
    train = [idx for idx in list(range(1, 10)) if not idx in test]
    args.datanum = 2
    args.chnum = 62
    args.sessionnum = 2
if args.data == 'BCI4_2a':
    test = list(map(int, args.test.split(',')))
    train = [idx for idx in list(range(1, 10)) if not idx in test]
    args.datanum = 4
    args.chnum = 22
    args.sessionnum = 2

args.train = train
net = model.Distribution(args).to(device)
args.hyper = [1, 0.5, 0.05, 0]
args.hyper_finetune = [1, 0, 0, 0]
# 交叉熵损失
loss = torch.nn.CrossEntropyLoss().to(device)
# Adam优化器
optimizer = torch.optim.Adam(net.parameters(), lr=args.learningrate)

if args.is_training == 'train':
    os.makedirs(args.checkpoint, exist_ok=True)
    # x_anc（锚点TS），x_pos（正样本TS），x_neg（负样本TS），标签（锚点的标签），x_anc_rest（锚点RS），x_pos_rest（正样本RS），x_neg_rest（负样本RS）
    # dataset = [] # Set dataset # return x_anc, x_pos, x_neg, label, x_anc_rest, x_pos_rest, x_neg_rest
    dataset = OpenBMI_RS_MI_Dataset(
        root_dir='openBMI_MI',  
        subjects=args.train,
        size=args.size,
        ch_num=args.chnum,
        is_training=True
    )
    data_tr = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=8, drop_last=True)
    data_val = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
    # data_val = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=8)
    manage = utils.find_best_model(args.checkpoint, args)
    manage.code_copy(os.path.join(args.checkpoint, 'run'))
    # 阶段1: 特征解耦训练
    for epoch in range(args.epoch):
        net.train()
        x, y, output, acc_epoch, loss_epoch, count_epoch = one_epoch_distribution(net, data_tr, optimizer, loss, args.hyper, args, train='train')
        print('Epoch  : ', epoch + 1, 'Acc: ', round(acc_epoch, 7), 'Loss: ', np.round(loss_epoch['Total'], 5))
        
        net.eval()
        with torch.no_grad():
            x, y, output, acc_epoch, loss_epoch, count_epoch = one_epoch_distribution(net, data_val, optimizer, loss, args.hyper, args, train='val')
            manage.update(net, args.checkpoint, epoch, acc_epoch, loss_epoch['CLS'])
            print('Val    : ', epoch + 1, 'Acc: ', round(acc_epoch, 7), 'Loss: ', np.round(loss_epoch['Total'], 5))

    manage.training_finish(net, args.checkpoint)

elif args.is_training == 'test':
    manage = utils.test_model(args.checkpoint, args, 'result')
    dataset = [] # Set dataset # return x_anc, x_pos, x_neg, label, x_anc_rest, x_pos_rest, x_neg_rest
    # 返回目标被试的静息态信号（即RS信号），用于生成伪TS信号
    # dataset_RS = [] # Set dataset # return x_anc_rest
    dataset = OpenBMI_RS_MI_Dataset(
        root_dir='openBMI_MI',  
        subjects=args.train,
        size=args.size,
        ch_num=args.chnum,
        is_training=False
    )
    dataset_RS = OpenBMI_RSOnly_Dataset(
        root_dir='openBMI_MI',
        subject_id=test[0],
        size=args.size,
        ch_num=args.chnum,
    )
    restore = torch.load(os.path.join(args.checkpoint, 'model-best.pth'), map_location=torch.device('cpu'))
    net.load_state_dict(restore, strict=True)

    for session in range(args.sessionnum):
        print(f"\n=== 开始 Session {session+1}/2 ===")
        torch.cuda.empty_cache()
        net.load_state_dict(restore, strict=True)
        restore_before = net.state_dict()
        # 阶段 2: 使用静息态生成伪TS
        dataset_RS.current_session = session  # 设置当前 session
        dataset_RS.is_training = True         # 设置为 RS-only 模式
        data_test_RS = DataLoader(dataset_RS, batch_size=1, shuffle=False, num_workers=8)
        print("🚀 阶段2: 伪TS生成中...")
        RS, RS_update, RS_update_label = generate_signal_from_RS(net, data_test_RS, loss, args.hyper, args, train='train')
        RS_update = torch.cat(RS_update, dim=0)
        RS = torch.cat(RS, dim=0)
        RS_update_label = torch.cat(RS_update_label, dim=0)
        net.load_state_dict(restore_before, strict=True)

        dataset_RS.testset[session]['x'] = torch.cat([RS, RS_update], dim=3) # 拼接 RS + 伪TS
        # print(f"📐 拼接后 testset x shape: {dataset_RS.testset[session]['x'].shape}")
        dataset_RS.testset[session]['y'] = RS_update_label # 伪标签
        dataset_RS.is_training = False  # 切换为测试数据模式
        dataset_RS.current_session = session

        net.train()
        print("🧪 阶段3: 微调中...")
        data_test_RS = DataLoader(dataset_RS, batch_size=args.batch, shuffle=True, num_workers=8)
        net.load_state_dict(restore_before)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learningrate/10 )
        kd_loss = torch.nn.KLDivLoss().to(args.device)
        for epoch in range(10):
            args.cdist = 0
            # 阶段3: RS EEG校准微调（模型适应）
            x, y, output, acc_epoch, loss_epoch, count_epoch = one_epoch_distribution(net, data_test_RS, optimizer, kd_loss, args.hyper_finetune, args, train='RS')
            # print(f"[Debug] RS input shape: {x.shape}")
            print(f"→ Finetune Epoch {epoch+1}/10: Acc={round(acc_epoch, 5)}")

        # 阶段 4: 测试
        print("📊 阶段4: 最终测试评估")
        # dataset.is_training = 'test'
        # dataset.finetuning = False
        data_test = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
        net.eval()
        with torch.no_grad():
            x, y, output, acc_epoch, loss_epoch, count_epoch = one_epoch_distribution(net, data_test, optimizer, loss, args.hyper, args, train='val')
            print(str(test[0])+'S'+str(session)+': ', count_epoch, 'Acc: ', round(acc_epoch, 7), 'Loss: ', round(loss_epoch['Total'], 7))
            manage.total_result(str(test[0])+'S'+str(session), count_epoch, round(acc_epoch, 7), round(loss_epoch['Total'], 7))