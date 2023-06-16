from myimports import *
import AN_prep

AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
AlexNet_model.classifier[4] = nn.Linear(4096, 1024)
AlexNet_model.classifier[6] = nn.Linear(1024, 2)
print(AlexNet_model.eval())

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AlexNet_model.to(device)

# Loss
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.SGD(AlexNet_model.parameters(), lr = 0.001, momentum=0.9)

patience = 5
best_val_loss = float('inf')
no_improvement = 0

for epoch in range(15):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(AN_prep.train_data_loader['train']):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = AlexNet_model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
     # evaluate the model on the validation set
    val_loss = 0.0
    with torch.no_grad():
        for data in AN_prep.valid_data_loader['valid']:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = AlexNet_model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # calculate the average validation loss
    val_loss /= len(AN_prep.valid_data_loader['valid'])

    # check if the validation loss has improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improvement = 0
        # save the best model
        torch.save(AlexNet_model.state_dict(), 'best_model.pt')
    else:
        no_improvement += 1
        # check if the training process should be stopped
        if no_improvement >= patience:
            print('Early stopping after %d epochs without improvement' % patience)
            break

print('Finished Training of AlexNet')

correct = 0
total = 0
with torch.no_grad():
    for data in AN_prep.test_data_loader['test']:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = AlexNet_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))