from myimports import *
import AN_prep
import dataset as ds

# Using AlexNet model
alexnet = models.alexnet(weights = True)


''' keeping the core model intact, so that we cen
    leverage prior feature map training, & replace
    the image classifier layers with
    output layers for our model.'''
for param in alexnet.parameters():
    param.requires_grad = False
    # gradient will not be computed during backpropagation

alexnet.classifier[6] = nn.Linear(4096, 2)
alexnet.classifier.add_module('7', nn.LogSoftmax(dim = 1))

# Training hyperparameters
params = OrderedDict(hidden_units = [256, 512],
                     dropout = [0.4, 0.5],
                     num_classes = [4],
                     lr = [0.001],
                     batch_size = [16],
                     n_epochs = [100])


''' this class manages the run and epoch parameter initializations
    to track trianing outputs and feed output data to TensorBoard'''
class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # training
def train_func(model, train_data_loader, train_data_size, epochs = 25):
    m = RunBuilder() 
    # loss function
    loss_func = nn.NLLLoss()
    # optimizer
    optimizer = optim.Adam(alexnet.parameters(), lr = run.lr)
    for run in RunBuilder.get_runs(params):
        alexnet = models.AlexNet(run.hidden_units, run.dropout, run.num_classes)
        optimizer = optim.Adam(alexnet.parameters(), lr = run.lr)

    m.begin_run(run, alexnet, AN_prep.train_data_loader)

    start = time.time()
    history = []
    best_acc = 0.0

    for epoch in range(run.n_epochs):
        m.begin_epoch()
        print('Epoch: {}/{}'.format(epoch+1, epochs))

        #train
        # model.train()

        # loss and accuracy within epoch
        train_loss = 0.0
        train_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # clean existing gradients
            optimizer.zero_grad()

            # forward pass - compute outputs oninput data using the model
            outputs = model(inputs)

            # compute loss
            loss = loss_func(outputs, labels)

            # Backpropagate
            loss.backward()

            # update parameters
            optimizer.step()

            # compute the total loss for the batch and add it to train_loss
            train_loss += loss.item()*inputs.size(0)

            # compute accuracy
            ret, predicitons = torch.max(outputs.data, 1)
            correct_counts = predicitons.eq(labels.data.view_as(predicitons))
            # correct_counts to floats
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)
        
            with torch.no_grad():
                model.eval()

            avg_train_loss = train_loss/train_data_size
            avg_train_acc = train_acc/train_data_size
        m.epoch_end()
        history.append([avg_train_loss, avg_train_acc])
        # epoch_end = time.time()
        print('Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%'.format(epoch+1, loss.item(), acc.item()*100))
    m.end_run()
    return model, history


def predict(model, test_image_name):
    transform = AN_prep.image_transforms['test']
    test_image = Image.open(test_image_name)
    plt.imshow(test_image)

    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()

        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(2, dim = 1)
        for i in range(2):
            print('Prediction', i+1, ':', AN_prep.idx_to_class[topclass.numpy()[0][i]])


# initalize alexnet
num_epochs = 100
print(summary(alexnet, (3, 224, 224)))
trained_model, history = train_func(alexnet, AN_prep.train_data_loader, AN_prep.train_data_size, num_epochs)
torch.save(history, ds.dataset+'history.pt')