"""Train a model on SQuAD.

Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util

from args import get_train_args
from collections import OrderedDict
from json import dumps
from models import BiDAF
from models2 import BiDAF2
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD


def main(args):
    # Set up logging and devices
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)

    # =============== Get model =================
    
    log.info('Building model...')
    model1 = BiDAF2(word_vectors=word_vectors,
                  hidden_size=args.hidden_size,
                  drop_prob=args.drop_prob)
    model1 = nn.DataParallel(model1, args.gpu_ids)
    if args.load_path1:
        log.info(f'Loading checkpoint from {args.load_path1}...')
        model1, step = util.load_model(model1, args.load_path1, args.gpu_ids)
    else:
        step = 0
    model1 = model1.to(device)
    model1.train()
    ema1 = util.EMA(model1, args.ema_decay)
    
    # =============== Get model =================
    
    
    
    # =============== Get model =================
    
    log.info('Building model...')
    model3 = BiDAF(word_vectors=word_vectors,
                  hidden_size=args.hidden_size,
                  drop_prob=args.drop_prob)
    model3 = nn.DataParallel(model3, args.gpu_ids)
    if args.load_path3:
        log.info(f'Loading checkpoint from {args.load_path3}...')
        model3, step = util.load_model(model3, args.load_path3, args.gpu_ids)
    else:
        step = 0
    model3 = model3.to(device)
    model3.train()
    ema3 = util.EMA(model3, args.ema_decay)
    
    # =============== Get model =================
    
    # =============== Get model =================
    
    log.info('Building model...')
    model2 = BiDAF(word_vectors=word_vectors,
                  hidden_size=args.hidden_size,
                  drop_prob=args.drop_prob)
    model2 = nn.DataParallel(model2, args.gpu_ids)
    if args.load_path2:
        log.info(f'Loading checkpoint from {args.load_path2}...')
        model2, step = util.load_model(model2, args.load_path2, args.gpu_ids)
    else:
        step = 0
    model2 = model2.to(device)
    model2.train()
    ema2 = util.EMA(model2, args.ema_decay)
    
    # =============== Get model =================
    
    # ============ mask ===============
    
    vocab_size = word_vectors.shape[0] + 1

    # ============ mask ===============
    

    # Get saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer1 = optim.Adadelta(model1.parameters(), args.lr,
                               weight_decay=0.0001)
    
    optimizer2 = optim.Adam(model2.parameters(), 0.001,
                               weight_decay=0.0001)
    
    optimizer3 = optim.Adadelta(model3.parameters(), args.lr,
                               weight_decay=0.0001)
    
    scheduler1 = sched.LambdaLR(optimizer1, lambda s: 1.)  # Constant LR
    scheduler2 = sched.LambdaLR(optimizer2, lambda s: 1.)  # Constant LR
    scheduler3 = sched.LambdaLR(optimizer3, lambda s: 1.)  # Constant LR

    # Get data loader
    log.info('Building dataset...')
    train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = SQuAD(args.dev_record_file, args.use_squad_v2)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn)

    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    
    
    # Evaluate and save checkpoint
    log.info(f'Evaluating at step {step}...')
    ema1.assign(model1)
    ema2.assign(model2)
    ema3.assign(model3)
    results, pred_dict = evaluate(model1, model2, model3, dev_loader, device,
                                    args.dev_eval_file,
                                    args.max_ans_len,
                                    args.use_squad_v2)
    
    ema1.resume(model1)
    ema2.resume(model2)
    ema3.resume(model3)

    # Log to console
    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
    log.info(f'Dev {results_str}')

    # Log to TensorBoard
    log.info('Visualizing in TensorBoard...')
    for k, v in results.items():
        tbx.add_scalar(f'dev/{k}', v, step)
    util.visualize(tbx,
                    pred_dict=pred_dict,
                    eval_path=args.dev_eval_file,
                    step=step,
                    split='dev',
                    num_visuals=args.num_visuals)
    
    while epoch != args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
                # Setup for forward
                
                cw_idxs = cw_idxs.to(device)
                qw_idxs = qw_idxs.to(device)
                cc_idxs = cc_idxs.to(device)
                qc_idxs = qc_idxs.to(device)
                batch_size = cw_idxs.size(0)
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()

                # Forward
                log_p1_1, log_p2_1 = model1(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
                log_p1_2, log_p2_2 = model2(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
                log_p1_3, log_p2_3 = model3(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
                y1, y2 = y1.to(device), y2.to(device)

                log_p1, _ = torch.max(torch.stack([log_p1_1, log_p1_2, log_p1_3]), dim=0)
                log_p2, _ = torch.max(torch.stack([log_p2_1, log_p2_2, log_p2_3]), dim=0)
                
                y1, y2 = y1.to(device), y2.to(device)
                
                loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model1.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(model2.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(model3.parameters(), args.max_grad_norm)
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
                scheduler1.step(step // batch_size)
                scheduler2.step(step // batch_size)
                scheduler3.step(step // batch_size)
                ema1(model1, step // batch_size)
                ema2(model2, step // batch_size)
                ema3(model3, step // batch_size)

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         NLL=loss_val)
                tbx.add_scalar('train/NLL', loss_val, step)
                # tbx.add_scalar('train/LR',
                #                optimizer.param_groups[0]['lr'],
                #                step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema1.assign(model1)
                    ema2.assign(model2)
                    ema3.assign(model3)
                    results, pred_dict = evaluate(model1, model2, model3, dev_loader, device,
                                                  args.dev_eval_file,
                                                  args.max_ans_len,
                                                  args.use_squad_v2)
                    # saver.save(step, model, results[args.metric_name], device)
                    ema1.resume(model1)
                    ema2.resume(model2)
                    ema3.resume(model3)

                    # Log to console
                    results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar(f'dev/{k}', v, step)
                    util.visualize(tbx,
                                   pred_dict=pred_dict,
                                   eval_path=args.dev_eval_file,
                                   step=step,
                                   split='dev',
                                   num_visuals=args.num_visuals)


def evaluate(model1, model2, model3, data_loader, device, eval_file, max_len, use_squad_v2):
    nll_meter = util.AverageMeter()

    model1.eval()
    model2.eval()
    model3.eval()
    pred_dict = {}
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            log_p1_1, log_p2_1 = model1(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
            log_p1_2, log_p2_2 = model2(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
            log_p1_3, log_p2_3 = model3(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
            y1, y2 = y1.to(device), y2.to(device)
            
            log_p1, _ = torch.max(torch.stack([log_p1_1, log_p1_2, log_p1_3]), dim=0)
            log_p2, _ = torch.max(torch.stack([log_p2_1, log_p2_2, log_p2_3]), dim=0)
            
            loss = F.nll_loss(log_p1, y1) + F.nll_loss(log_p2, y2)
            nll_meter.update(loss.item(), batch_size)

            # Get F1 and EM scores
            p1, p2 = log_p1.exp(), log_p2.exp()
            starts, ends = util.discretize(p1, p2, max_len, use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(NLL=nll_meter.avg)

            preds, _ = util.convert_tokens(gold_dict,
                                           ids.tolist(),
                                           starts.tolist(),
                                           ends.tolist(),
                                           use_squad_v2)
            pred_dict.update(preds)

    model1.train()
    model2.train()
    model3.train()

    results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
    results_list = [('NLL', nll_meter.avg),
                    ('F1', results['F1']),
                    ('EM', results['EM'])]
    if use_squad_v2:
        results_list.append(('AvNA', results['AvNA']))
    results = OrderedDict(results_list)

    return results, pred_dict


if __name__ == '__main__':
    main(get_train_args())
