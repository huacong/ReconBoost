import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self,input_dim = 512, output_dim = 40):
        super(Head,self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, output_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class MHead(nn.Module):
    def __init__(self, input_dim = 128, output_dim = 3):
        super(MHead, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = self.fc(x)
        return x
    

class DynamicNet(object):
    def __init__(self, c0, lr, common_space = False, dataset = "MN40", n_class = 40):
        self.models = []
        self.models_name = []
        self.dataset = dataset
        self.c0 = c0
        self.lr = lr
        self.boost_rate  = nn.Parameter(torch.tensor(lr, requires_grad=True, device="cuda"))
        
        self.common_space = common_space
        self.num_classes = n_class

        if common_space:
            if self.dataset == 'MSA':
                self.head = MHead().cuda()
            else:
                self.head = Head(output_dim=self.num_classes).cuda()
        
        
    def add(self, model,model_name):
        item = -1
        for i in range(0,len(self.models_name)):
            if self.models_name[i] == model_name:
                item = i
                break

        if item != -1:
            self.models.pop(item)
            self.models_name.pop(item)
            
        self.models.append(model)
        self.models_name.append(model_name)

    def parameters(self):
        params = []
        if self.common_space:   
            for m in self.models:
                for name, param in m.named_parameters():
                        if not 'fc' in name: 
                            params.append(param)
            params.extend(self.head.parameters())
        else:
            for m in self.models:
                params.extend(m.parameters())
        
        params.append(self.boost_rate)

        return params

    def zero_grad(self):
        for m in self.models:
            m.zero_grad()

    def to_cuda(self):
        for m in self.models:
            m.cuda()

    def to_eval(self):
        for m in self.models:
            m.eval()
        if self.common_space:
            self.head.eval()
            
        
    def to_train(self):
        for m in self.models:
            m.train(True)
        if self.common_space:
            self.head.train(True)
            
    def forward(self, data, mask_model = None):
        if len(self.models) == 0:
            return self.c0

        prediction = None

        if self.dataset == "CREMAD" or self.dataset == 'AVE':
            spec, image = data
            model_input_map = {
                'audio': spec,
                'visual': image
            }
        elif self.dataset == 'MView40':
            img_1, img_2 = data
            model_input_map = {
                'img_1': img_1,
                'img_2': img_2
            }
        elif self.dataset == 'MSA':
            vision, audio, text = data
            model_input_map = {
                'text': text,
                'audio': audio,
                'visual': vision
            }

        with torch.no_grad():
            for i in range(0,len(self.models)):
                m = self.models[i]
                m_name = self.models_name[i]
                if m_name == mask_model:
                    continue

                pred = m(model_input_map[m_name])
                if prediction is None:
                    prediction = pred
                else: 
                    prediction = prediction + pred

        return self.boost_rate * prediction

    def forward_grad(self, data):
        prediction = None

        if self.dataset == "CREMAD" or self.dataset == 'AVE':
            spec, image = data
            model_input_map = {
                'audio': spec,
                'visual': image
            }
        elif self.dataset == 'MView40':
            img_1, img_2 = data
            model_input_map = {
                'img_1': img_1,
                'img_2': img_2
            }
        elif self.dataset == 'MSA':
            vision, audio, text = data
            model_input_map = {
                'text': text,
                'audio': audio,
                'visual': vision
            }
        
        for i in range(0,len(self.models)):
            m = self.models[i]
            m_name = self.models_name[i]
            pred = m(model_input_map[m_name])

            if prediction is None:
                prediction = pred
            else: 
                prediction = prediction + pred

        return self.boost_rate * prediction
      

    def get_model_name(self):
        return self.models_name
    
    def to_file(self, path):
        if self.common_space:
            d = {'models_name': self.models_name, 'common_space':True, 'c0': self.c0, 'lr': self.lr, 'boost_rate': self.boost_rate,'head': self.head.state_dict()}
        else:
            d = {'models_name': self.models_name, 'common_space':False, 'c0': self.c0, 'lr': self.lr, 'boost_rate': self.boost_rate, }
        torch.save(d, path)
        print('The ensemble model has been saved at {}'.format(path))
