import torch
import torch.nn as nn
import copy

class LinearWithConstraint(nn.Linear):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

class EEGNet(nn.Module):
    def __init__(self, args, global_f, feature_size):
        super(EEGNet, self).__init__()
        self.args = args
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 8, [125, 1], [1, 1], padding=[62, 0]),
            nn.BatchNorm2d(8),
            Conv2dWithConstraint(8, 16, [1, self.args.chnum], [1, 1], padding=[0, 0], groups=8, max_norm=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d([4, 1], [4, 1]),
            nn.Dropout2d(0.25)
        )
        # 这在输入 [B, 16, 187, 1] 的情况下，输出 [B, 16, 23, 1]，Flatten 后就是 368，是预期值
        # self.encoder2 = nn.Sequential(
        #     nn.Conv2d(16, 16, [16, 1], [1, 1], padding=[8, 0], groups = 16),
        #     # nn.Conv2d(32, 32, [1, 1], [1, 1], padding=[8, 0], groups=1),
        #     nn.BatchNorm2d(16),
        #     nn.ELU(),
        #     nn.AvgPool2d([8, 1], [8, 1]),
        #     nn.Dropout2d(0.25)
        # )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 16, [15, 1], [1, 1], padding=[0, 0], groups=16),  # 14 → 1
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d([1, 30], [1, 30]),  # 22 → 22//30 = 0 → 再改为 22 → 23 要调合适的kernel
            nn.AdaptiveAvgPool2d((1, 23)),
            nn.Dropout2d(0.25)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(global_f, args.datanum, bias=True)
    def forward(self, x):
        raise NotImplementedError

class Distribution(nn.Module):
    def __init__(self, args):
        super(Distribution, self).__init__()
        self.args = args
        if args.model == 'EEGNet':
            global_f = 368
            feature_size=64
            model = EEGNet(args, global_f = global_f, feature_size=feature_size)
            # 主干网络 (对应E_stem)
            self.encoder1 = model.encoder1
            # 特征解耦模块  对应 E_task 和 E_subject
            self.decomposer = Decomposer(16) # number of channel

            #for subject-dependent feature
            # 对于主体依赖的特征
            self.encoder_sub = model.encoder2 # E_subject
            self.encoder_sub2 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(global_f, feature_size)
            )

            #for task-dependent feature
            # 对于任务依赖的特征
            self.encoder2= copy.deepcopy(self.encoder_sub) # E_task
            self.classifier = model.classifier # 分类器 C
            self.flatten = nn.Flatten()

            #for task distribution
            # 论文中的类别原型 P_y
            self.mu = torch.nn.Parameter(torch.randn((args.datanum, 1, 16)))
            # 分布参数
            self.sigma = torch.nn.Parameter(torch.randn((args.datanum, 1, 16)))

        elif args.model == 'Conformer':
            from conformer import PatchEmbedding, TransformerEncoder, ClassificationHead
            from einops.layers.torch import Rearrange
            self.patch = PatchEmbedding(emb_size=40, ch=args.chnum) # ShallowNet
            self.decomposer = Decomposer(40)
            self.rearrange1 = Rearrange('b hw e -> b e hw 1')
            self.rearrange2 = Rearrange('b e (h) (w) -> b (h w) e')

            ##for task-dependent feature
            self.encoder = TransformerEncoder(depth=6, emb_size=40)
            self.classifier= self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(40 * 44, 256),
                nn.ELU(),
                nn.Dropout(0.5),
                nn.Linear(256, 32),
                nn.ELU(),
                nn.Dropout(0.3),
                nn.Linear(32, args.datanum)
            )

            ##for subject-dependent feature
            self.encoder_sub = nn.Sequential(
                    nn.Conv2d(40, 40, [3, 1], [1, 1], padding=[1, 0], groups = 40),
                    nn.BatchNorm2d(40),
                    nn.ELU(),
                    nn.AvgPool2d([8, 1], [8, 1]),
                    nn.Dropout2d(0.25)
                )
            self.encoder_sub2 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(200, 64)
            )

            # for task distribution
            if args.multiproto == 1:
                self.mu = torch.nn.Parameter(torch.randn((args.datanum, 1, 40)))
                self.sigma = torch.nn.Parameter(torch.randn((args.datanum, 1, 40)))
            else:
                self.mu = torch.nn.Parameter(torch.normal(mean=torch.zeros(args.datanum, args.multiproto, 40), std=torch.ones(args.datanum, args.multiproto, 40)*0.0001))
                self.sigma = torch.nn.Parameter(torch.normal(mean=torch.zeros(args.datanum, args.multiproto, 40), std=torch.ones(args.datanum, args.multiproto, 40)*0.0001))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)


    def forward(self, x, train=None):
        if self.args.model == 'EEGNet':
            # print("x:", x.shape)
            feature = self.encoder1(x)
            # print("feature after encoder1:", feature.shape)
            sdf, tdf = self.decomposer(feature)
            # print("sdf before encoder_sub:", sdf.shape)
            sdf = self.encoder_sub(sdf)
            # print("sdf after encoder_sub", sdf.shape)
            sdf = self.encoder_sub2(sdf)
            # print("sdf after encoder_sub2:", sdf.shape)
            tdf = self.encoder2(tdf)
            output = self.classifier(self.flatten(tdf))
            # 返回分类结果、任务依赖特征和主体依赖特征
            return output, tdf, sdf

        elif self.args.model == 'Conformer':
            feature = self.patch(x.permute(0, 1, 3, 2))
            feature = self.rearrange1(feature)
            sdf, tdf = self.decomposer(feature)
            sdf = self.encoder_sub(sdf)
            sdf = self.encoder_sub2(sdf)
            tdf = self.rearrange2(tdf)
            tdf = self.encoder(tdf)
            output = self.classifier(tdf)

            return output, tdf.permute(0, 2, 1).unsqueeze(-1), sdf

class Decomposer(nn.Module):
    def __init__(self, nfeat):
        super(Decomposer, self).__init__()
        self.nfeat = nfeat
        self.embed_layer = nn.Sequential(nn.Conv2d(nfeat, nfeat*2, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(nfeat*2), nn.ELU(), nn.Dropout())

    def forward(self, x):
        embedded = self.embed_layer(x)
        rele, irre = torch.split(embedded, [int(self.nfeat), int(self.nfeat)], dim=1)

        return rele, irre



class CORAL(nn.Module):
    """
    CORAL 损失函数的实现。
    """
    def __init__(self):
        super(CORAL, self).__init__()

    def forward(self, source, target):
        d = source.size(1)

        if source.size(0) < 2 or target.size(0) < 2:
            return torch.tensor(0.0, device=source.device)

        source_mean = torch.mean(source, dim=0, keepdim=True)
        source_centered = source - source_mean
        cov_source = (source_centered.t() @ source_centered) / (source.size(0) - 1)

        target_mean = torch.mean(target, dim=0, keepdim=True)
        target_centered = target - target_mean
        cov_target = (target_centered.t() @ target_centered) / (target.size(0) - 1)

        loss = torch.sum((cov_source - cov_target)**2) / (4 * d**2)
        return loss


if __name__ == '__main__':
    x = torch.rand((1, 1, 750, 3))
    model = EEGNet(368, 64)
    out = model(x)
