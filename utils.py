import numpy as np 

def load_data():
    '''loads data in format:
    "reviewer: [timestamp, target_grade, target_decision, [features]]
    features are GPA, SAT, ...
    target_grade is the rating which was given to the student by this reviewer
    target_decision is if the student was actually admitted  
    '''
    read_dictionary = np.load('../admissions.npy',allow_pickle='TRUE').item()
    return read_dictionary

def normalize(anchor):
    norm_anch = anchor.squeeze()
    norm_anch -= 2*norm_anch.min(0, keepdim=True)[0]
    norm_anch /= norm_anch.max(0, keepdim=True)[0]
    return norm_anch -1

def validate(data, valid_keys, target_net, label="validation"):
    '''sends unseen data into the target net and observes the average validation reward on this data'''
    target_net.eval()
    cum_reward = 0
    num_reviews = 0 
    for reviewer in data:
        if reviewer not in valid_keys:
            continue
        for review_session in data[reviewer]:
            for idx, student in enumerate(review_session):
                
                timestamp, target_decision, final_decision, features = student
                target_decision = int(target_decision>1)
                features = torch.Tensor(features).to(device).unsqueeze(0)

                output = target_net(features)
                action = output.max(1)[1].view(1, 1)
                cum_reward += torch.tensor([reward(action, target_decision)], device=device).item()
                num_reviews+=1
    print("average ", label, " reward: ", cum_reward/num_reviews)

def students_by_year(data, keys, return_len_students=False):
    all_students = []
    for reviewer in data:
        cum_reward = 0
        number_reviews = 0
        if reviewer not in keys:
            continue
        hidden_size=1
            
        for review_session in data[reviewer]:
            for idx, student in enumerate(review_session):
                
                timestamp, target_decision, final_decision, features, svm_decision, svm_confidence = student
                all_students.append(student)
    all_students = np.array(all_students)
    min_timestamp = min(all_students[:,0])
    sorted_students = np.array(sorted(all_students, key=lambda x: x[0]))
    year = ((np.array(sorted_students)[:,0]-min(np.array(sorted_students)[:,0]))/31557600).astype(int)
    clustered_reviews_by_year = np.split(sorted_students, np.unique(year, return_index=True)[1][1:])
    clustered_reviews_by_year = np.array(clustered_reviews_by_year)
    return clustered_reviews_by_year, len(sorted_students) if return_len_students else clustered_reviews_by_year

def all_students_sorted_by_year(data, keys):
    all_students = []
    for reviewer in data:
        cum_reward = 0
        number_reviews = 0
        if reviewer not in keys:
            continue
        hidden_size=1
            
        for review_session in data[reviewer]:
            for idx, student in enumerate(review_session):
                
                timestamp, target_decision, final_decision, features, svm_decision, svm_confidence = student
                all_students.append(student)
    all_students = np.array(all_students)
    min_timestamp = min(all_students[:,0])
    sorted_students = np.array(sorted(all_students, key=lambda x: x[0]))
    year = ((np.array(sorted_students)[:,0]-min(np.array(sorted_students)[:,0]))/31557600).astype(int)
    sorted_students[:,0] = year
    return sorted_students