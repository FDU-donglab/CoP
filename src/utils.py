import torch
import random
import os
import torch.distributed as dist
import numpy as np
from scipy import interpolate
from skimage import io, color, util
import json
from tqdm import tqdm
def setup_for_distributed(is_master):
    """禁用非主进程的打印输出"""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    # 在 init_distributed_mode 开头添加
    print(f"RANK: {os.environ.get('RANK', 'NOT SET')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'NOT SET')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'NOT SET')}")
    # 场景1: 通过 torch.distributed.launch 或 torchrun 启动
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    
    # 场景2: Slurm 集群启动
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.gpu = int(os.environ['SLURM_LOCALID'])  
    
    # 场景3: 单卡模式
    elif torch.cuda.is_available():
        print('Running on single GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    
    else:
        raise RuntimeError("GPU not available, distributed training not supported")

    # 关键顺序：先绑定设备，再初始化进程组
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank
    )
    
    
    print(f"LOCAL_RANK: {args.local_rank}, RANK: {args.rank}, WORLD_SIZE: {args.world_size}")
    setup_for_distributed(args.rank == 0)
    
class ElasticWeightConsolidation:
    def __init__(self, model, dataloader, device, lambda_ewc):
        print("Initializing EWC for Incremental Learning, this will take a few minutes (about 1 epoch)...")
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()
        print("Initializing finished.")

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = torch.zeros_like(p)

        self.model.eval()
        progress_bar = tqdm(self.dataloader, desc='Calculating Fisher', unit='batch', ncols=150)
        for terms in progress_bar:
            inputs, labels, masks_in = terms['input'].to(self.device), terms['target'].to(self.device), terms['mask']
            self.model.zero_grad()
            outputs, masks = self.model(inputs, masks_in)
            loss = torch.mean(torch.abs(outputs - labels) ** 2 * masks)
            loss.backward()

            for n, p in self.params.items():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataloader)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._precision_matrices:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return self.lambda_ewc * loss
    
class ElasticWeightConsolidation_S_2:
    def __init__(self, model_encoder,model_decoder, dataloader, device, lambda_ewc):
        print("Initializing EWC for Incremental Learning, this will take a few minutes (about 1 epoch)...")
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self.dataloader = dataloader
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()
        print("Initializing finished.")

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = torch.zeros_like(p)

        self.model_encoder.eval()
        self.model_decoder.eval()
        progress_bar = tqdm(self.dataloader, desc='Calculating Fisher', unit='batch', ncols=150)
        for terms in progress_bar:
            inputs, labels, masks_in = terms['input'].to(self.device), terms['target'].to(self.device), terms['mask']
            self.model.zero_grad()
            output_features, _ = self.model_encoder(inputs, masks_in)
            outputs = self.model_decoder(output_features)
            loss = torch.mean(torch.abs(outputs - labels) ** 2)
            loss.backward()

            for n, p in self.params.items():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataloader)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._precision_matrices:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return self.lambda_ewc * loss

def validate_stage_func(allowed_strings):
    def decorator(func):
        def wrapper(train_stage):
            if train_stage in allowed_strings:
                return func(train_stage)
            else:
                print("Please input {}".format(allowed_strings))
        return wrapper
    return decorator

def config_to_json(args, output_dir, stage_name="stage_1"):
    config_dict = args
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, '{}_config.json'.format(stage_name))
    # Save the dictionary to a JSON file
    with open(output_file, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f'Configurations saved to {output_file}')

def json_to_config(input_file):
    # Load the dictionary from the JSON file
    with open(input_file, 'r') as f:
        config_dict = json.load(f)

    return config_dict

def freeze_model_parameters(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False

def setup_seed(seed=2022):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_patch_ext(output_dir, file_name, patch, tag = 'result' , ext = 'jpg'):
    output_path = os.path.join(output_dir, f"{file_name[:-4]}_{tag}.{ext}")
    # skimage io.imsave needs image data in range 0-1 when saving as png
    # patch = util.img_as_ubyte(patch)
    io.imsave(output_path, patch)

def model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6}M")
    print(f"Trainable parameters: {trainable_params/1e6}M")

def load_checkpoint_encoder_only(checkpoint_load_path, model):
    print(f">>>>>>>>>> Resuming from {checkpoint_load_path} ..........")
    if checkpoint_load_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_load_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(checkpoint_load_path, map_location='cpu')
    if 'model' in checkpoint:
        encoder_keys = {k: v for k, v in checkpoint['model'].items() if k in checkpoint['model'] and k.startswith('encoder.')}
    else: encoder_keys = {k: v for k, v in checkpoint.items() if k in checkpoint and k.startswith('encoder.')}
    
    model_state_dict = model.state_dict()
    model_state_dict.update(encoder_keys)
    msg = model.load_state_dict(model_state_dict, strict=False)
    print(msg)
    del checkpoint, model_state_dict
    torch.cuda.empty_cache()

def load_encoder_weights(model, encoder_state_dict):
    model_state_dict = model.state_dict()
    encoder_keys = {k: v for k, v in encoder_state_dict.items() if k in model_state_dict and k.startswith('encoder.')}
    model_state_dict.update(encoder_keys)
    model.load_state_dict(model_state_dict)

def load_checkpoint(checkpoint_load_path, model, optimizer, lr_scheduler):
    print(f">>>>>>>>>> Resuming from {checkpoint_load_path} ..........")
    if checkpoint_load_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_load_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(checkpoint_load_path, map_location='cpu')
    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=True)
    else: msg = model.load_state_dict(checkpoint, strict=True)
    print(msg)
    if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print(f"=> loaded successfully '{checkpoint_load_path}' (epoch {checkpoint['epoch']})")
    del checkpoint
    torch.cuda.empty_cache()

def load_checkpoint_only(checkpoint_load_path, model):
    print(f">>>>>>>>>> Resuming from {checkpoint_load_path} ..........")
    if checkpoint_load_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_load_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(checkpoint_load_path, map_location='cpu')
    # 检查是否有'module.'前缀
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    # 检查是否有'module.'前缀
    if any(k.startswith('module.') for k in state_dict.keys()):
        # 去除'module.'前缀
        new_state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        msg = model.load_state_dict(new_state_dict, strict=True)
    else:
        msg = model.load_state_dict(state_dict, strict=True)
    print(msg)
    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir, logger):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    logger.info(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        logger.info(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def load_pretrained(config, model, logger):
    logger.info(f">>>>>>>>>> Fine-tuned from {config.PRETRAINED} ..........")
    checkpoint = torch.load(config.PRETRAINED, map_location='cpu')
    checkpoint_model = checkpoint['model']
    
    if any([True if 'encoder.' in k else False for k in checkpoint_model.keys()]):
        checkpoint_model = {k.replace('encoder.', ''): v for k, v in checkpoint_model.items() if k.startswith('encoder.')}
        logger.info('Detect pre-trained model, remove [encoder.] prefix.')
    else:
        logger.info('Detect non-pre-trained model, pass without doing anything.')

    if config.MODEL.TYPE == 'swin':
        logger.info(f">>>>>>>>>> Remapping pre-trained keys for SWIN ..........")
        checkpoint = remap_pretrained_keys_swin(model, checkpoint_model, logger)
    elif config.MODEL.TYPE == 'vit':
        logger.info(f">>>>>>>>>> Remapping pre-trained keys for VIT ..........")
        checkpoint = remap_pretrained_keys_vit(model, checkpoint_model, logger)
    else:
        raise NotImplementedError

    msg = model.load_state_dict(checkpoint_model, strict=False)
    logger.info(msg)
    
    del checkpoint
    torch.cuda.empty_cache()
    logger.info(f">>>>>>>>>> loaded successfully '{config.PRETRAINED}'")
    

def remap_pretrained_keys_swin(model, checkpoint_model, logger):
    state_dict = model.state_dict()
    
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                logger.info(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    logger.info(f"{key}: Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1 ** 0.5)
                    dst_size = int(L2 ** 0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    logger.info("Original positions = %s" % str(x))
                    logger.info("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                            relative_position_bias_table_pretrained.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model


def remap_pretrained_keys_vit(model, checkpoint_model, logger):
    # Duplicate shared rel_pos_bias to each layer
    if getattr(model, 'use_rel_pos_bias', False) and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
        logger.info("Expand the shared relative position embedding to each transformer block.")
    num_layers = model.get_num_layers()
    rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
    for i in range(num_layers):
        checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()
    checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")
    
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                logger.info("Position interpolate for %s from %dx%d to %dx%d" % (key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                logger.info("Original positions = %s" % str(x))
                logger.info("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias
    
    return checkpoint_model