import torch 
import torch.nn as nn 
import numpy as np 
from tqdm import tqdm 
import time
import torch.utils.data as data
from polish_sign_dataset import Polish_dataset 
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class SLRGRU(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(SLRGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, bidirectional =True, batch_first=True, dropout=drop_prob  )
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.relu = nn.ReLU()    
        # self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, bidirectional =True, batch_first=True, dropout=drop_prob  )

    def forward(self, x, h):
        
        out, h = self.gru(x, h)
        # out, h = self.lstm(x, h)

        out = self.fc(out[:,-1]) 
        # print(out)
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_()
        return hidden
    
    


def train(train_loader, learn_rate,hidden_dim=64, EPOCHS=50, model_type="GRU",input_dim = 258,output_dim =65):
    
    # Setting common hyperparameters
    # input_dim = next(iter(train_loader))[0].shape[2]
    
    n_layers = 2
    # Instantiating the models
    if model_type == "GRU":
        model = SLRGRU(input_dim, hidden_dim, output_dim, n_layers)
    
    
    # Defining loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    
    # model.eval()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    # Start training loop

    for epoch in tqdm(range(1,EPOCHS+1)):
        start_time = time.time()
        avg_loss = 0.
        counter = 0
        h = model.init_hidden(train_loader.batch_size)
        
        correct =0
    
        all_y = []
        all_y_pred = []
        
       
        for idx,(x, label ,id) in enumerate(train_loader):

            # print(idx)
            # print(x.shape)
            counter += 1
            if model_type == "GRU":
                h = h.data
            # else:
            #     h = tuple([e.data for e in h])
            model.zero_grad()
            # print(label)
            
            out, h = model(x.float(), h)
            loss = criterion(out, label)
            
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            y_pred = out.max(1, keepdim=True)[1]
            all_y.extend(label)
            all_y_pred.extend(y_pred)
            # correct += (out == label).float().sum()
        all_y = torch.stack(all_y, dim=0)
        all_y_pred = torch.stack(all_y_pred, dim=0).squeeze()
        
        top1acc = accuracy_score(all_y, all_y_pred)*100
        print("ejhfejh",top1acc)
        #     if counter%200 == 0:
        #         print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
        # current_time = time.time()
        # print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss/len(train_loader)))
        # print("Total Time Elapsed: {} seconds".format(str(current_time-start_time)))
        # epoch_times.append(current_time-start_time)
        # accuracy = 100 * correct / len(trainset)
        # print("Accuracy = {}".format(accuracy))
    torch.save(model.state_dict(), 'model_e{EPOCHS}.pth')
    return model
    
    
    
    






def test(testloader , model):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels ,video_id = data
            # calculate outputs by running images through the network
            h = model.init_hidden(testloader.batch_size)
            print(h.shape)
            print(testloader.batch_size)
            print(images.shape)
            outputs ,h = model(images.float() ,h)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the  test images: %d %%' % (
        100 * correct / total))























# gru_model = train(train_loader, lr, model_type="GRU")


if __name__ == '__main__':
    
    full_dataset = Polish_dataset(index_file_path='/home/user/Documents/projet/floderwlasl/WLASL/data/splits/asl100.json', 
                           split=['train','val'], pose_root="video__landmarks",img_transforms=None, video_transforms=None, num_samples= 50)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    
    
    
    print(len(train_dataset))
    print(len(test_dataset))
    # exit(0)

    
    
    
    train_data_loader = data.DataLoader(dataset=train_dataset ,batch_size=64,  
                                                    shuffle=True ,drop_last=True)
    # print(len(train_data_loader))
    test_data_loader = data.DataLoader(dataset=test_dataset ,batch_size=64,  
                                                    shuffle=True ,drop_last=True)
    
    
    
    lr = 0.001

    gru_model = train(train_data_loader, lr,EPOCHS=50 ,model_type="GRU")
    
    test(test_data_loader ,gru_model )
