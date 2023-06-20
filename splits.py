from myimports import os, train_test_split, shutil

foc_fol_path = r"\foc_unfoc\focused"
train_foc_fol = r"\al_net\dataset\training\focused" # has focused training data
test_foc_fol = r"\al_net\dataset\testing\focused" # has focused testing data
val_foc_fol = r'\al_net\dataset\validation\focused'

unfoc_fol_path = r"\foc_unfoc\unfocused"
train_unfoc_fol = r"\al_net\dataset\training\unfocused" # has unfocused training data
test_unfoc_fol = r"\al_net\dataset\testing\unfocused" # has unfocused testing data
val_unfoc_fol = r"\al_net\dataset\validation\unfocused"

try:
    if not os.path.exists(val_foc_fol):
        os.makedirs(val_foc_fol)
except Exception as e:
    print('------------failed in dir creation')

    print(e)

#           focused, train->focused, testing->focused, focused
def splits_method(folder_path, folder_train, folder_test, folder_valid, name):
    train_data = {}
    test_data = {}
    valid_data = {}

    # list out all the images
    folder_images = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]

    # train test split 
    folder_train, folder_test = train_test_split(folder_images, test_size = 0.2, random_state = 42)
    folder_train, folder_valid = train_test_split(folder_train, test_size=0.1, random_state = 42)
    #store train and test splits in dictionaries
    train_data[name] = folder_train
    test_data[name] = folder_test
    valid_data[name] = folder_valid

    print(name, '\n train: ', train_data[name][:10], '\n\n')
    print(name, '\n test:', test_data[name][:10], '\n\n')
    print(name, '\n valid:', valid_data[name][:10], '\n\n')

    print('suxxezz')
    return folder_train, folder_test, folder_valid

# need to store the images that are split in their corresponding folders

def store(folder_train, train_path, folder_test, test_path, folder_valid, valid_path, name):
    # storing training data
    for image_path in folder_train:    
        # Extracting the filename from the image path
        image_filename = os.path.basename(image_path)
        # Create the new path for the image in the new folder
        new_image_path = os.path.join(train_path, image_filename)
        # Copy the image file to the new folder
        shutil.copy2(image_path, new_image_path)

    for image_path in folder_test:    
    # Extracting the filename from the image path
        image_filename = os.path.basename(image_path)
        # Create the new path for the image in the new folder
        new_image_path = os.path.join(test_path, image_filename)
        # Copy the image file to the new folder
        shutil.copy2(image_path, new_image_path)

    for image_path in folder_valid:    
    # Extracting the filename from the image path
        image_filename = os.path.basename(image_path)
        # Create the new path for the image in the new folder
        new_image_path = os.path.join(valid_path, image_filename)
        # Copy the image file to the new folder
        shutil.copy2(image_path, new_image_path)
    print('success')

train_data, test_data, valid_data = splits_method(foc_fol_path, train_foc_fol, test_foc_fol, val_foc_fol,name='focused')
print(len(train_data), len(test_data), len(valid_data))
store(train_data, train_foc_fol, test_data, test_foc_fol, valid_data, val_foc_fol,  'focused') # focused training images loaded in needed folder

train_data, test_data, valid_data = splits_method(unfoc_fol_path, train_unfoc_fol, test_unfoc_fol, val_unfoc_fol, name='unfocused')
print(len(train_data), len(test_data), len(valid_data))
store(train_data, train_unfoc_fol, test_data, test_unfoc_fol, valid_data, val_unfoc_fol, 'unfocused') # unfocused training images loaded in needed folder
