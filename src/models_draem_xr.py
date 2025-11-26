import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        return x1, x2, x3, x4


class UNetDecoder(nn.Module):
    def __init__(self, base_ch: int = 32, out_ch: int = 1) -> None:
        super().__init__()
        self.up1 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up3 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ) -> torch.Tensor:
        d1 = self.up1(x4)
        d1 = self.dec1(torch.cat([d1, x3], dim=1))
        d2 = self.up2(d1)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))
        d3 = self.up3(d2)
        d3 = self.dec3(torch.cat([d3, x1], dim=1))
        out = self.out_conv(d3)
        return out


class ReconstructiveSubnet(nn.Module):
    """Approximation der Nested-UNet-Rekonstruktion."""

    def __init__(self, in_ch: int = 1) -> None:
        super().__init__()
        self.encoder = UNetEncoder(in_ch=in_ch)
        self.decoder = UNetDecoder(out_ch=in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4 = self.encoder(x)
        out = self.decoder(x1, x2, x3, x4)
        return out


class DiscriminativeSubnet(nn.Module):
    """U-Net, welches eine Anomaliekarte erzeugt."""

    def __init__(self, in_ch: int = 2) -> None:
        super().__init__()
        self.encoder = UNetEncoder(in_ch=in_ch)
        self.decoder = UNetDecoder(out_ch=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4 = self.encoder(x)
        out = self.decoder(x1, x2, x3, x4)
        return torch.sigmoid(out)


class DRAEM_XR(nn.Module):
    """
    Komplette Architektur:
    - reconstructive_subnet: rekonstruiert anomaly-free Image
    - discriminative_subnet: liefert Anomalie-Heatmap
    """

    def __init__(self, in_ch: int = 1) -> None:
        super().__init__()
        self.reconstructive = ReconstructiveSubnet(in_ch=in_ch)
        self.discriminative = DiscriminativeSubnet(in_ch=in_ch * 2)

    def forward(
        self, img_anom: torch.Tensor, img_orig: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recon = self.reconstructive(img_anom)
        disc_in = torch.cat([recon, img_orig], dim=1)
        segm = self.discriminative(disc_in)
        return recon, segm
