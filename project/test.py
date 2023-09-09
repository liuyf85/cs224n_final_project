# import torch
# import torch.nn as nn

# # 定义字符嵌入和字符级别的卷积
# vocab_size = 1000  # 字典中字符的数量
# embedding_dim = 4  # 嵌入向量的维度
# conv_filters = 5  # 卷积核的数量
# conv_kernel_size = 1  # 卷积核的大小
# char_embedding = nn.Embedding(vocab_size, embedding_dim)
# char_conv = nn.Conv1d(in_channels=embedding_dim, out_channels=conv_filters, kernel_size=conv_kernel_size)

# # 假设输入是一个字符序列，形状为(batch_size, seq_len, num_chars)
# batch_size = 2
# seq_len = 4
# num_chars = 2

# input_seq = torch.arange(16).reshape([2, 4, 2])

# input_seq = input_seq.reshape([batch_size * seq_len, num_chars])
# # 使用字符嵌入层进行嵌入
# embedded_seq = char_embedding(input_seq).permute([0, 2, 1])

# print(embedded_seq.shape)


# # 使用字符级别的卷积
# conv_seq = char_conv(embedded_seq)

# conv_seq = conv_seq.permute([0, 2, 1])


# conv_seq = conv_seq.reshape([batch_size, seq_len, num_chars, -1])

# # for name, param in char_conv.named_parameters():
# #     print(f'{name}\n{param}\n\n')

# print(conv_seq.shape)

# conv_seq, _ = torch.max(conv_seq, dim = 2)

# print(conv_seq.shape) 

# # 输出卷积序列的形状
# # print(conv_seq.shape)

# # import torch

# # x = torch.Tensor([[[111, 112], [121, 122]],[[211, 212 ], [221, 222]]])

# # print(x.reshape([4, 2]).reshape([2, 2, 2]))

# import torch

# x = torch.arange(24).reshape([4, 6])

# y = torch.rand([4, 6]) < 0.5

# print(x)

# print(x[:, torch.sum(y, dim = -1)])     


# # ============== prob distribution ===============
    
#     # Get data loader
#     log.info('Building dataset...')
#     train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
#     train_loader = data.DataLoader(train_dataset,
#                                    batch_size=args.batch_size,
#                                    shuffle=False,
#                                    num_workers=args.num_workers,
#                                    collate_fn=collate_fn)
#     dev_dataset = SQuAD(args.dev_record_file, args.use_squad_v2)
#     dev_loader = data.DataLoader(dev_dataset,
#                                  batch_size=args.batch_size,
#                                  shuffle=False,
#                                  num_workers=args.num_workers,
#                                  collate_fn=collate_fn)   
    
#     num_test = 0
    
#     for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in train_loader:
#         # Setup for forward
        
#         num_test = num_test + 1
#         if num_test > 15 +12 + 1 : exit()
#         if num_test != 1 and num_test != (15 + 1) and num_test != (15 + 12 + 1) : continue
        
#         cw_idxs = cw_idxs.to(device)
#         qw_idxs = qw_idxs.to(device)
#         cc_idxs = cc_idxs.to(device)
#         qc_idxs = qc_idxs.to(device)

#         # Forward
#         log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
        
#         b, l = log_p1.shape
        
#         import matplotlib.pyplot as plt
        
#         x = torch.arange(l)
        
#         plt.clf()
#         plt.bar(x.cpu(), log_p1.detach().squeeze().cpu())
#         plt.savefig(f'{num_test}_char_mask_start.jpg')
        
#         plt.clf()
#         plt.bar(x.cpu(), log_p2.detach().squeeze().cpu())
#         plt.savefig(f'{num_test}_char_mask_end.jpg')
        
        
     
    
#     # ============== prob distribution ===============

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
    ema = util.EMA(model1, args.ema_decay)
    
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
    ema = util.EMA(model2, args.ema_decay)
    
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
    ema = util.EMA(model3, args.ema_decay)
    
    # =============== Get model =================