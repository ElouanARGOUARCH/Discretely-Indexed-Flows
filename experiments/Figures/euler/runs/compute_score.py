import torch
list_score_nll = []
list_score_dkl = []
number_runs = 20
lines = 256
columns = 197
import numpy
for i in range(number_runs):
    filename = 'euler_dif' + str(i) + '.sav'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dif = torch.load(filename,map_location=torch.device(device))
    list_score_nll.append(torch.tensor([dif.loss_values[-1]]))
    hist_target_samples, x_edges, y_edges = numpy.histogram2d(dif.target_samples[:, 1].numpy(),
                                                              dif.target_samples[:, 0].numpy(), bins=(lines, columns),
                                                              range=[[0, columns], [0, lines]], normed = True)
    hist_target_samples = torch.tensor(hist_target_samples)
    with torch.no_grad():
        dif_samples = dif.sample_model(dif.target_samples.shape[0])
    hist_dif_samples, x_edges, y_edges = numpy.histogram2d(dif_samples[:, 1].numpy(),
                                                              dif.target_samples[:, 0].numpy(), bins=(lines, columns),
                                                              range=[[0, columns], [0, lines]], normed = True)
    hist_dif_samples = torch.tensor(hist_dif_samples)
    DKL_hist = torch.sum(torch.nan_to_num(hist_target_samples*torch.log(hist_target_samples)) - torch.nan_to_num(hist_dif_samples*torch.log(hist_dif_samples)))
    list_score_dkl.append(torch.tensor([DKL_hist]))

print('mean score NLL = ' + str(torch.mean(torch.cat(list_score_nll), dim = 0).item()))
print('std NLL= ' + str(torch.std(torch.cat(list_score_nll), dim = 0).item()))
print('mean score DKL= ' + str(torch.mean(torch.cat(list_score_dkl), dim = 0).item()))
print('std DKL = ' + str(torch.std(torch.cat(list_score_dkl), dim = 0).item()))
