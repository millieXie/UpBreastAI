import torch
from Model.RUnet import RUnet

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model_component(path, part='encoder', freeze=False):
    model = RUnet(num_cls=1)
    ckpt = torch.load(path, map_location=DEVICE)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)

    if part == 'encoder':
        module = model.ADC_encoder.to(DEVICE)
    elif part == 'decoder':
        module = model.decoder_fuse.to(DEVICE)
    else:
        raise ValueError(f"Unknown part type: {part}")

    if freeze:
        for p in module.parameters():
            p.requires_grad = False

    module.eval()
    return module


def DCE_encoder():
    return load_model_component(
        r"/DCE_model.pth",
        part='encoder',
        freeze=False
    )

def DWI_encoder():
    return load_model_component(
        r"/DWI_model.pth",
        part='encoder'
    )

def ADC_encoder():
    return load_model_component(
        r"/ADC_model.pth",
        part='encoder'
    )

def DCE_decoder():
    return load_model_component(
        r"/DCE_model.pth",
        part='decoder'
    )

def DWI_decoder():
    return load_model_component(
        r"/DWI_model.pth",
        part='decoder'
    )

def ADC_decoder():
    return load_model_component(
        r"/ADC_model.pth",
        part='decoder'
    )
