import torch

# input:tensor
# output:int/float
def get_entropy_of_dataset(tensor:torch.Tensor):
    """Calculate the entropy of the entire dataset"""
    label_column = [t[-1].tolist() for t in tensor]     
    def CalculateEntropy(probs):
        return (-torch.sum(probs*torch.log2(probs)))    
    set_label=list(set(label_column))                  
    counts_probs=[]
    for x in set_label:
        counts_probs.append(label_column.count(x)/len(label_column))    
    k = (CalculateEntropy(torch.tensor(counts_probs))).item()           
    return k
    pass


def get_avg_info_of_attribute(tensor:torch.Tensor, attribute:int):
    """Return avg_info of the attribute provided as parameter"""
    Current_col_attribute = [t[attribute].tolist() for t in tensor]     
    multiple_diff_class = list(set(Current_col_attribute))              
    label_column = [t[-1].tolist() for t in tensor]                    
    total_length_of_column=len(Current_col_attribute)
    o=0
    for x in multiple_diff_class:
        feature_count = Current_col_attribute.count(x)          
        mul_factor_probs = torch.tensor(feature_count/total_length_of_column)
        t1=[]
        t2=[]
        for i in range(len(Current_col_attribute)):
            if Current_col_attribute[i]==x:
                t1.append(x)                     # t1 contains column attributes which a unique in each iteration
                t2.append(label_column[i])                      # t2 contains the corresponding target column which is mapped to the attribute column
        new_tensor = torch.cat((torch.tensor(t1).unsqueeze(1),torch.tensor(t2).unsqueeze(1)),dim=1)     
        test_entropy = get_entropy_of_dataset(new_tensor)
        if torch.isnan(torch.tensor(test_entropy))==False:
            o+=(mul_factor_probs*test_entropy).item()     
    return o
    pass

def get_information_gain(tensor:torch.Tensor, attribute:int):
    """Return Information Gain of the attribute provided as parameter"""
    return (torch.round(torch.tensor(get_entropy_of_dataset(tensor)) - get_avg_info_of_attribute(tensor,attribute),decimals=4)).item() 
    pass


def get_selected_attribute(tensor:torch.Tensor):
    """
    Return a tuple with the first element as a dictionary which has IG of all columns
    and the second element as an integer representing attribute number of selected attribute

    example : ({0: 0.123, 1: 0.768, 2: 1.23} , 2)
    """
    gainInfo_dictionary={}
    for i in range(len(tensor[0]) - 1):
        gainInfo_dictionary[i]=get_information_gain(tensor,i)           
    
    max_gain = max(gainInfo_dictionary.values())
    
    for i in gainInfo_dictionary.keys():
        if gainInfo_dictionary[i] == max_gain:
            return (gainInfo_dictionary,int(i))                         
    
    return ({},-1)
    pass