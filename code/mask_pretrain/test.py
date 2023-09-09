import torch

def mask_and_predict(mask_prob):

    input_tensor = torch.rand([3, 3])

    mask_indices = torch.rand(input_tensor.shape) < mask_prob
    masked_input_tensor = input_tensor.masked_fill(mask_indices, 100)
    
    # print(input_tensor)
    # print(masked_input_tensor)

    last_hidden_state = torch.rand([3, 3, 16])
    
    masked_hidden_state = last_hidden_state.squeeze()[mask_indices.squeeze()]
    
    print(mask_indices.squeeze())
    
    print(masked_hidden_state)

mask_and_predict(0.15)

x = torch.rand([2, 2, 4])

a = torch.rand(x.shape[:-1]) < 0.5


# import torch 
# import torch.nn.functional as F

# pred = torch.rand([4, 10])


# pred = F.softmax(pred, dim = -1)

# print(pred)

# label = torch.randint(low = 0, high = 9, size = [4])
# print(label)

# loss = F.nll_loss(pred, label)

# print(loss)