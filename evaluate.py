import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, criterion):
    net.eval()
    num_val_batches = len(dataloader)
    total_loss = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(
                device=device,
                dtype=torch.float32
            ).unsqueeze(1)

            # predict the mask
            mask_pred = net(image)

            loss = criterion(mask_pred, mask_true)
            total_loss += loss.item()

    net.train()
    return total_loss / len(dataloader)
