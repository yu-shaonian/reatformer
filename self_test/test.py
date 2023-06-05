import torch
depth_predictions_all = []
for i in range(30):
    depth_predictions_all.append((torch.rand(2,128,160)-0.5)*400)
depth_values = torch.rand(2,192)*192

B,H,W = depth_predictions_all[0].shape
for i,depth_pred_all in enumerate(depth_predictions_all):
    value_all = []
    for b in range(B):
        # import ipdb
        # ipdb.set_trace()
        depth_pred = depth_pred_all[b,:,:]
        depth_value = depth_values[b,:].squeeze(0)
        mask_left = depth_pred < 0
        mask_right = depth_pred >= 192
        mask_inside = torch.logical_and(depth_pred >= 0, depth_pred < 191)
        # 注意mask对应的是true false
        value = torch.zeros_like(depth_pred)
        inside_index = depth_pred[mask_inside]
        left_index = torch.floor(inside_index).long()
        right_index = left_index + 1
        delta = inside_index - torch.floor(inside_index)

        value[mask_inside] = depth_value[left_index] * (1 - delta) + depth_value[right_index] * delta
        value[mask_left] = depth_value[0]
        value[mask_right] = depth_value[-1]
        value_all.append(value.unsqueeze(0))
    depth_predictions_all[i] = torch.cat(value_all,dim=0)

print("test")