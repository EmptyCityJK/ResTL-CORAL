import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import warnings
import swanlab
import random
import model
import utils
from train import one_epoch_distribution, generate_signal_from_RS
from Dataset import OpenBMI_RS_MI_Dataset, OpenBMI_RSOnly_Dataset

warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 实验参数配置
parser = argparse.ArgumentParser(description="Cross-subject")
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-te", "--test", type=str, default='1', help='要作为测试集的被试编号')
parser.add_argument("-m", "--model", type=str, default='EEGNET')
parser.add_argument("-e", "--epoch", type=int, default=100)
parser.add_argument("-l", "--learningrate", type=float, default=0.0005)
parser.add_argument("-t", "--is_training", type=str, default='train', choices=['train', 'test'])
parser.add_argument("-b", "--batch", type=int, default=128)
parser.add_argument("-d", "--data", type=str, default='OpenBMI', choices=['BCI4_2b', 'BCI4_2a', 'OpenBMI'])
parser.add_argument("--size", type=int, default=750)
parser.add_argument("--rest", type=bool, default=False)
parser.add_argument("--cdist", type=float, default=1e-5)
parser.add_argument("--checkpoint", type=str, default='checkpoint')
parser.add_argument('--multiproto', type=int, default=1)
parser.add_argument('--lambda_coral', type=float, default=0.5, help='Weight for the CORAL loss')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

torch.multiprocessing.set_sharing_strategy('file_system')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device
# 将命令行传入的字符串test ID转为整数列表
test = list(map(int, args.test.split(',')))

# --- 数据集参数配置 ---
if args.data == 'BCI4_2b':
    train = [idx for idx in list(range(1, 10)) if not idx in test]
    args.datanum = 2
    args.chnum = 3
    args.sessionnum = 5
if args.data == 'OpenBMI':
    train = [idx for idx in list(range(1, 5)) if not idx in test]
    args.datanum = 2
    args.chnum = 62
    args.sessionnum = 2
if args.data == 'BCI4_2a':
    train = [idx for idx in list(range(1, 10)) if not idx in test]
    args.datanum = 4
    args.chnum = 22
    args.sessionnum = 2

args.train = train
net = model.Distribution(args).to(device)
args.hyper = [1, 0.5, 0.05, 0]
args.hyper_finetune = [1, 0, 0, 0]

# --- 损失函数和优化器 ---
cls_loss_fn = torch.nn.CrossEntropyLoss().to(device)
coral_loss_fn = model.CORAL().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.learningrate)

# ================================== 训练阶段 ==================================
if args.is_training == 'train':
    # SwanLab 初始化
    swanlab.init(
        project="da",
        workspace="weiguai",
        config={
            "learning_rate": args.learningrate,
            "architecture": args.model,
            "dataset": args.data,
            "epochs": args.epoch,
            "batch_size": args.batch,
            "lambda_coral": args.lambda_coral
        }
    )
    os.makedirs(args.checkpoint, exist_ok=True)
    manage = utils.find_best_model(args.checkpoint, args)
    manage.code_copy(os.path.join(args.checkpoint, 'run'))

    # 1. 准备数据加载器
    print("准备源域数据加载器 (Source Loader)...")
    source_dataset = OpenBMI_RS_MI_Dataset(
        root_dir='openBMI_MI',
        subjects=args.train,
        size=args.size,
        ch_num=args.chnum,
        is_training=True
    )
    source_loader = DataLoader(source_dataset, batch_size=args.batch, shuffle=True, num_workers=8, drop_last=True)

    print("准备目标域数据加载器 (Target Loader)...")
    target_dataset = OpenBMI_RS_MI_Dataset(
        root_dir='openBMI_MI',
        subjects=test,
        size=args.size,
        ch_num=args.chnum,
        is_training=True
    )
    target_loader = DataLoader(target_dataset, batch_size=args.batch, shuffle=True, num_workers=8, drop_last=True)

    data_val = DataLoader(source_dataset, batch_size=32, shuffle=False, num_workers=8)

    # 2. 开始领域自适应训练
    print("\n🚀 开始领域自适应训练 (含CORAL损失)...")
    for epoch in range(args.epoch):
        net.train()
        
        epoch_task_loss, epoch_coral_loss, epoch_total_loss = 0.0, 0.0, 0.0
        total_samples, correct_samples = 0, 0
        
        target_iter = iter(target_loader)
        
        for i, source_data in enumerate(source_loader):
            # --- 已修正: 使用字典键名来获取数据 ---
            x_src = source_data['x_anc'].to(device, dtype=torch.float)
            y_src = source_data['label'].argmax(dim=1).type(torch.LongTensor).to(device)

            try:
                target_data = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data = next(target_iter)
            
            # --- 已修正: 使用字典键名来获取数据 ---
            x_tgt = target_data['x_anc'].to(device, dtype=torch.float)

            if x_src.size(0) != x_tgt.size(0):
                continue

            optimizer.zero_grad()
            output_src, tdf_src, _ = net(x_src)
            _, tdf_tgt, _ = net(x_tgt)

            task_loss = cls_loss_fn(output_src, y_src)
            alignment_loss = coral_loss_fn(torch.flatten(tdf_src, start_dim=1), torch.flatten(tdf_tgt, start_dim=1))
            total_loss = task_loss + args.lambda_coral * alignment_loss
            
            total_loss.backward()
            optimizer.step()

            epoch_task_loss += task_loss.item()
            epoch_coral_loss += alignment_loss.item()
            epoch_total_loss += total_loss.item()
            
            _, predicted = torch.max(output_src.data, 1)
            total_samples += y_src.size(0)
            correct_samples += (predicted == y_src).sum().item()

        avg_acc = correct_samples / total_samples
        avg_total_loss = epoch_total_loss / len(source_loader)
        avg_task_loss = epoch_task_loss / len(source_loader)
        avg_coral_loss = epoch_coral_loss / len(source_loader)
        
        print(f"Epoch: {epoch + 1}/{args.epoch} | "
              f"Acc: {avg_acc:.4f} | "
              f"Total Loss: {avg_total_loss:.4f} | "
              f"Task Loss: {avg_task_loss:.4f} | "
              f"CORAL Loss: {avg_coral_loss:.4f}")

        # SwanLab 记录指标
        swanlab.log({
            "acc": avg_acc,
            "total_loss": avg_total_loss,
            "task_loss": avg_task_loss,
            "coral_loss": avg_coral_loss
        })

        # 3. 验证模型
        net.eval()
        with torch.no_grad():
            val_hyper = [1, 0, 0, 0]
            _, _, _, acc_epoch_val, loss_epoch_val, _ = one_epoch_distribution(
                net, data_val, optimizer, cls_loss_fn, val_hyper, args, train='val')
            manage.update(net, args.checkpoint, epoch, acc_epoch_val, loss_epoch_val['CLS'])
            print(f"Val Epoch: {epoch + 1} | Acc: {round(acc_epoch_val, 5)} | Loss: {np.round(loss_epoch_val['Total'], 5)}")

    manage.training_finish(net, args.checkpoint)
    # SwanLab 训练结束
    swanlab.finish()

# ================================== 测试阶段 ==================================
elif args.is_training == 'test':
    manage = utils.test_model(args.checkpoint, args, 'result')
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
    restore = torch.load(os.path.join(args.checkpoint, 'model-best2.pth'), map_location=torch.device('cpu'))
    net.load_state_dict(restore, strict=True)

    for session in range(args.sessionnum):
        print(f"\n=== 开始 Session {session+1}/{args.sessionnum} ===")
        torch.cuda.empty_cache()
        net.load_state_dict(restore, strict=True)
        restore_before = net.state_dict()
        
        # 阶段 2: 使用静息态生成伪TS
        dataset_RS.current_session = session
        dataset_RS.is_training = True
        data_test_RS = DataLoader(dataset_RS, batch_size=1, shuffle=False, num_workers=8)
        print("🚀 阶段2: 伪TS生成中...")
        RS, RS_update, RS_update_label = generate_signal_from_RS(net, data_test_RS, cls_loss_fn, args.hyper, args, train='train')
        RS_update = torch.cat(RS_update, dim=0)
        RS = torch.cat(RS, dim=0)
        RS_update_label = torch.cat(RS_update_label, dim=0)
        net.load_state_dict(restore_before, strict=True)

        dataset_RS.testset[session]['x'] = torch.cat([RS, RS_update], dim=3)
        dataset_RS.testset[session]['y'] = RS_update_label
        dataset_RS.is_training = False
        dataset_RS.current_session = session

        # 阶段 3: 微调
        net.train()
        print("🧪 阶段3: 微调中...")
        data_finetune = DataLoader(dataset_RS, batch_size=args.batch, shuffle=True, num_workers=8)
        net.load_state_dict(restore_before)
        optimizer_finetune = torch.optim.Adam(net.parameters(), lr=args.learningrate / 10)
        kd_loss = torch.nn.KLDivLoss().to(args.device)
        for epoch in range(10):
            args.cdist = 0
            _, _, _, acc_epoch, _, _ = one_epoch_distribution(
                net, data_finetune, optimizer_finetune, kd_loss, args.hyper_finetune, args, train='RS')
            print(f"→ Finetune Epoch {epoch+1}/10: Acc={round(acc_epoch, 5)}")

        # 阶段 4: 测试
        print("📊 阶段4: 最终测试评估")
        data_test = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
        net.eval()
        with torch.no_grad():
            _, _, _, acc_epoch_test, loss_epoch_test, count_epoch = one_epoch_distribution(
                net, data_test, optimizer, cls_loss_fn, args.hyper, args, train='val')
            result_str = f"{test[0]}S{session}: {count_epoch} | Acc: {round(acc_epoch_test, 7)} | Loss: {round(loss_epoch_test['Total'], 7)}"
            print(result_str)
            manage.total_result(f"{test[0]}S{session}", count_epoch, round(acc_epoch_test, 7), round(loss_epoch_test['Total'], 7))