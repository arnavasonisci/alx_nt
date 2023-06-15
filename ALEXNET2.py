from myimports import *
import AN_prep
import dataset as ds

# load alexnet model with pre trained weights
alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

# Update the second classifier
alexnet.classifier[4] = nn.Linear(4096, 1024)
alexnet.classifier[6] = nn.Linear(1024, 2)
alexnet.classifier.add_module('7', nn.LogSoftmax(dim = 1))
print(alexnet)

#Instantiating CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# loss
loss_criterion = nn.NLLLoss()

# optimizer
optimizer = optim.Adam(alexnet.parameters())

# training of AlexNet
for epoch in range(100):
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(AN_prep.train_data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # output
        output = alexnet(inputs)
        # loss
        loss = loss_criterion(output, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 16 == 15:    # print every 16 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 16))
            running_loss = 0.0

print('Finished Training of AlexNet')

# Testing Accuracy
correct = 0
total = 0
with torch.no_grad():
    pass # replace and continue coding from here