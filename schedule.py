import random

schedule_list = []

fixed_mode = True
random_mode = False
guided_mode = False
greedy_mode = False
guided_mode_min_loss = False

#### baseline according to the stage
def schedule_model(stage, dataset):
    if random_mode:
        seed = random.randint(0,100)
        if dataset == 'MN40':
            if seed % 3 == 0:
               return "pt"
            elif seed % 3 == 1:
                return "img"
            else:
                return "mesh"
        elif dataset == "CREMAD" or dataset == 'AVE':
            if stage == 0:
                return "audio"
            elif stage == 1:
                return "visual"
            else:
                if seed % 2 == 0:
                    return "audio"
                else:
                    return "visual"
            
    if fixed_mode:
        if dataset == 'MN40' or dataset == 'MN40_simple':
            if stage % 3 == 0:
                return "pt"
            elif stage % 3 == 1:
                return "img"
            else:
                return "mesh"
            
        elif dataset == 'MOSEI':
            if stage % 3 == 0:
                return "text"
            elif stage % 3 == 1:
                return "visual"
            elif stage % 3 == 2:
                return "audio"
    
        elif dataset == "CREMAD" or dataset == 'AVE' or dataset == 'MOSEI_RAW':
            if stage % 2 == 0:
                #return "visual"
                return "audio"
            else:
                return "visual"
                # return "audio"
        elif dataset == 'MView40':
            if stage % 2 == 0:
                return "img_1"
            else:
                return "img_2"
            