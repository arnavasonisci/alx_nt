from myimports import *
import dataset as ds
# transform the data

image_transforms = {
        'train': transforms.Compose([
        transforms.RandomResizedCrop(size = 256, scale = (0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size = 224), # do we need this? the cells are not in the center of the frame.
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
        'test': transforms.Compose([
            transforms.Resize(size = 256), 
            transforms.CenterCrop(size = 224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
        'valid': transforms.Compose([
            transforms.Resize(size = 256), 
            transforms.CenterCrop(size = 224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    }

def classIdentification(dataset):
    # training, testing and validation paths
    train_directory = os.path.join(dataset, 'training')
    test_directory = os.path.join(dataset, 'testing')
    valid_directory = os.path.join(dataset, 'validation')

    bs = 16 # batch size

    data = {
        'train': datasets.ImageFolder(root = train_directory, transform = image_transforms['train']),
        'test': datasets.ImageFolder(root = test_directory, transform = image_transforms['test']),
        'valid': datasets.ImageFolder(root = valid_directory, transform = image_transforms['valid'])
    }

    idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
    print(idx_to_class)

    train_data_size = len(data['train'])
    print(train_data_size)
    test_data_size = len(data['test'])
    valid_data_size = len(data['valid'])

    train_data_loader = { 'train': DataLoader(data['train'], batch_size = bs, shuffle = True)}
    test_data_loader = {'test': DataLoader(data['test'], batch_size = bs, shuffle = False)}
    valid_data_loader = {'valid': DataLoader(data['valid'], batch_size=bs, shuffle=False)}
    return train_data_loader, train_data_size, test_data_loader, test_data_size, valid_data_loader, valid_data_size, idx_to_class

train_data_loader, train_data_size, test_data_loader, test_data_size, valid_data_loader, valid_data_size, idx_to_class = classIdentification(ds.dataset)
print(type(train_data_loader))  # it is a dictionary
print(len(train_data_loader['train']))