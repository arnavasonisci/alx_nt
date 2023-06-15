''' For each folder (focused, unfocused),
    I need to make a train and test split.
'''
from myimports import *
import ALEXNET




# # initalize alexnet
# num_epochs = 5
# print(summary(ALEXNET.alexnet, (3, 224, 224)))
# trained_model, history = ALEXNET.train_func(ALEXNET.alexnet, AN_prep.train_data_loader, AN_prep.train_data_size, loss_func, optimizer, num_epochs)
# torch.save(history, ds.dataset+'history.pt')

ALEXNET.predict(ALEXNET.trained_model, r"F:\Projects\MyLab\al_net\dataset\testing\focused\10.jpg")