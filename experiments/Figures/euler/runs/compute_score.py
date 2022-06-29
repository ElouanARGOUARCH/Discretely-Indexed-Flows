import torch
list_score = []
number_runs = 10
for i in range(number_runs):
    filename = 'euler_dif' + str(i) + '.sav'
    dif = torch.load(filename)
    list_score.append(torch.tensor([dif.loss_values[-1]]))

print('mean score = ' + str(torch.mean(torch.cat(list_score), dim = 0).item()))
print('std = ' + str(torch.std(torch.cat(list_score), dim = 0).item()))
