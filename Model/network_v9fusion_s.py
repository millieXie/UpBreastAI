
import torch.nn as nn
import torch.nn.functional as F
from Model.RUnet_encoder_decoder import *  

class CrossModalFuse(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.qkv = nn.Conv3d(channels * 3, channels * 3, kernel_size=1)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        q, k, v = self.qkv(x).chunk(3, dim=1)
        att = torch.softmax((q * k).sum(1, keepdim=True) / (q.size(1) ** 0.5), dim=1)
        out = self.proj(v * att)
        return out

class FusionBlock(nn.Module):
    def __init__(self, ch3=32, ch4=64, ch5=128, out_ch=128):
        super().__init__()
        self.conv_f3_1 = nn.Conv3d(ch3, 64, kernel_size=3, stride=2, padding=1)
        self.conv_f3_2 = nn.Conv3d(64, out_ch, kernel_size=3, stride=2, padding=1)
        self.conv_f4 = nn.Conv3d(ch4, out_ch, kernel_size=3, stride=2, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv3d(out_ch * 3, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x3, x4, x5):
        f3 = self.relu(self.conv_f3_1(x3))
        f3 = self.relu(self.conv_f3_2(f3))
        f4 = self.relu(self.conv_f4(x4))
        f5 = x5
        fused = torch.cat([f3, f4, f5], dim=1)
        out = self.fuse(fused)
        return out

class MemAE_SVDDHybrid(nn.Module):
    def __init__(self, in_channels=3, memory_dim=128, memory_size=500,
                 alpha_init=1.0, temperature=0.05):
        super().__init__()
        self.temperature = temperature

        # Encoder / Decoder
        self.DWI_extracter = DWI_encoder()
        self.ADC_extracter = ADC_encoder()
        self.DCE_extracter = DCE_encoder()

        self.DWI_decoder = DWI_decoder()
        self.ADC_decoder = ADC_decoder()
        self.DCE_decoder = DCE_decoder()

        self.DCE_fuse = FusionBlock()
        self.ADC_fuse = FusionBlock()
        self.DWI_fuse = FusionBlock()

        self.fuse5 = CrossModalFuse(128)

        self.memory5 = nn.Parameter(torch.randn(memory_size, memory_dim))
        nn.init.orthogonal_(self.memory5)
        self.alpha5 = nn.Parameter(torch.tensor(alpha_init))

        self.enc_proj5 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.2),
            nn.Conv3d(128, memory_dim, kernel_size=1, bias=False)
        )
        self.proj_back5 = nn.Conv3d(memory_dim, 128, kernel_size=1, bias=False)
        self.mem_norm = nn.LayerNorm(memory_dim)

        self.mem_query_proj5 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2)
        )

        self._proj_dec5 = nn.Conv3d(128, 128, 1, bias=False)
        nn.init.xavier_uniform_(self._proj_dec5.weight)

        self.register_buffer('center5', torch.zeros(memory_dim))
        self.nu = 0.05
        self.register_buffer('R', torch.tensor(0.0))

    def initialize_center(self, train_loader, device):
        self.eval()
        all_z = []
        print("-> Initializing SVDD center with normal data...")
        with torch.no_grad():
            for x, _ in train_loader:
                x = x.to(device).float()
                B, C, D, H, W = x.shape
                x_DCE, x_ADC, x_DWI = torch.split(x, 1, dim=1)

                DWI_f = self.DWI_fuse(*self.DWI_extracter(x_DWI)[2:])
                ADC_f = self.ADC_fuse(*self.ADC_extracter(x_ADC)[2:])
                DCE_f = self.DCE_fuse(*self.DCE_extracter(x_DCE)[2:])

                fuse5 = self.fuse5(DWI_f, ADC_f, DCE_f)
                z_proj5 = self.enc_proj5(fuse5)
                z_5 = F.adaptive_avg_pool3d(z_proj5, 1).view(B, -1)
                # 关键：不对 z_5 做 L2 归一化
                all_z.append(z_5.cpu())

        self.center5.data = torch.cat(all_z, dim=0).mean(dim=0).to(device)
        self.center5.data[torch.abs(self.center5.data) < 1e-6] = 0.1
        print(f"-> Center initialized. Mean value: {self.center5.data.mean().item():.4f}")

    def memory_inject(self, x, enc_proj, proj_back, memory, alpha):
        B, C, D, H, W = x.shape
        z = enc_proj(x)
        z_norm = F.normalize(z, dim=1)
        z_pooled = F.adaptive_avg_pool3d(z_norm, 1).view(B, -1)
        mem_norm = F.normalize(memory, dim=1)
        similarity = torch.matmul(z_pooled, mem_norm.t()) / self.temperature
        att_weights = F.softmax(similarity, dim=1)

        mem_read = torch.matmul(att_weights, memory).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        mem_read = F.interpolate(mem_read, size=(D, H, W), mode='trilinear', align_corners=False)
        mem_read_normed = self.mem_norm(mem_read.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        mem_read_proj = proj_back(mem_read_normed) * alpha
        return x + mem_read_proj, att_weights

    def memory_diversity_loss(self):
        mem = self.memory5
        n = mem.size(0)
        mem_norm = F.normalize(mem, dim=1)
        sim = torch.matmul(mem_norm, mem_norm.t())
        identity = torch.eye(n, device=mem.device)
        off_diag = sim - identity
        loss = torch.abs(off_diag).sum() / n
        return loss

    def forward(self, x):
        B, C, D, H, W = x.shape
        x_DCE, x_ADC, x_DWI = torch.split(x, 1, dim=1)

        # Encoder
        DWI_x1, DWI_x2, DWI_x3, DWI_x4, DWI_x5 = self.DWI_extracter(x_DWI)
        ADC_x1, ADC_x2, ADC_x3, ADC_x4, ADC_x5 = self.ADC_extracter(x_ADC)
        DCE_x1, DCE_x2, DCE_x3, DCE_x4, DCE_x5 = self.DCE_extracter(x_DCE)

        DWI_fuse = self.DWI_fuse(DWI_x3, DWI_x4, DWI_x5)
        ADC_fuse = self.ADC_fuse(ADC_x3, ADC_x4, ADC_x5)
        DCE_fuse = self.DCE_fuse(DCE_x3, DCE_x4, DCE_x5)
        fuse5 = self.fuse5(DWI_fuse, ADC_fuse, DCE_fuse)

        fuse5_mem, att_weights = self.memory_inject(fuse5, self.mem_query_proj5, self.proj_back5, self.memory5, self.alpha5)

        z_proj5 = self.enc_proj5(fuse5_mem)
        z_5 = F.adaptive_avg_pool3d(z_proj5, 1).view(B, -1)

        dist = torch.sum((z_5 - self.center5) ** 2, dim=1)

        target_shape = DWI_x5.shape[2:]
        fuse5_for_dec = F.interpolate(fuse5_mem, size=target_shape, mode='trilinear', align_corners=False)

        DWI_x5_mapped = self._proj_dec5(DWI_x5)
        ADC_x5_mapped = self._proj_dec5(ADC_x5)
        DCE_x5_mapped = self._proj_dec5(DCE_x5)

        DWI_x5_new = DWI_x5_mapped + fuse5_for_dec
        ADC_x5_new = ADC_x5_mapped + fuse5_for_dec
        DCE_x5_new = DCE_x5_mapped + fuse5_for_dec

        x_DWI_dec, _ = self.DWI_decoder(DWI_x1, DWI_x2, DWI_x3, DWI_x4, DWI_x5_new)
        x_ADC_dec, _ = self.ADC_decoder(ADC_x1, ADC_x2, ADC_x3, ADC_x4, ADC_x5_new)
        x_DCE_dec, _ = self.DCE_decoder(DCE_x1, DCE_x2, DCE_x3, DCE_x4, DCE_x5_new)

        x_rec = torch.cat([x_DCE_dec, x_ADC_dec, x_DWI_dec], dim=1)
        return x_rec, att_weights, z_5, dist

    def get_reconstruction_error(self, x):
        x_rec, _, _, _ = self.forward(x)
        error = F.mse_loss(x_rec, x, reduction='none')
        return error.reshape(error.shape[0], -1).mean(dim=1)
