import numpy as np 
import torch
import collections
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    norm_anch = norm_anch - norm_anch.min(0, keepdim=True)[0]
    norm_anch = norm_anch/norm_anch.max(0, keepdim=True)[0]
    return norm_anch 

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

def get_input_output_data(review_session, input_for_lstm):
    ### SVM Score for Student ###################################
    svm_decision = np.array(review_session)[:,-2]
    svm_decision = torch.Tensor(np.array(svm_decision, dtype=int))
    ### Reviewer Decisions for Students ####################################
    reviewer_decision = torch.Tensor(np.array(review_session)[:,1] > 1).to(device).to(torch.int64) 
    ### Previous Decisions Score for Student ####################################
    previous_decisions = torch.tensor(np.concatenate((np.array([0]), np.array(review_session)[1:,1] > 1)))
    lstm_input = transform_lstm_input(input_for_lstm, svm_decision, svm_decision)
    return lstm_input, reviewer_decision

def transform_lstm_input(input_for_lstm, svm_decision, previous_decisions):
    if input_for_lstm == "SVM+Decision":
        svm_decision, previous_decisions = torch.unsqueeze(svm_decision, 1), torch.unsqueeze(previous_decisions, 1)
        lstm_input = torch.cat((svm_decision, previous_decisions), dim=1)
        lstm_input = torch.unsqueeze(lstm_input, 0).to(device)
    elif input_for_lstm == "SVM":
        lstm_input = svm_decision
        lstm_input = torch.unsqueeze(torch.unsqueeze(lstm_input, 0), -1).to(device)
    elif input_for_lstm == "Decision":
        lstm_input = previous_decisions.to(torch.float)
        lstm_input = torch.unsqueeze(torch.unsqueeze(lstm_input, 0), -1).to(device)
    return lstm_input

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
    return clustered_reviews_by_year
    
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

def plot_n_decisions_vs_confidence(review_sessions, figname='./figures/resampled_confidence.png'):
    scoresByInterval = collections.defaultdict(list)
    for session in review_sessions:
        nSinceAccept = None
        for i in range(len(session)):
            timestamp, target_decision, final_decision, features, svm_decision, svm_confidence = session[i]
            accept = target_decision > 1
            svmScore = svm_confidence
            if accept:
                if nSinceAccept == None:
                    # Skip the first one
                    nSinceAccept = 0
                    continue
                binSinceAccept = str(nSinceAccept)
                if nSinceAccept > 5 and nSinceAccept <= 10:
                    binSinceAccept = "6-10"
                elif nSinceAccept > 10 and nSinceAccept <= 20:
                    binSinceAccept = "11-20"
                elif nSinceAccept > 20:
                    binSinceAccept = "> 20"
                scoresByInterval[binSinceAccept].append(svmScore)
                nSinceAccept = 0
            else:
                if nSinceAccept != None:
                    nSinceAccept += 1
    averageScoresByInterval = {}


    for inter in scoresByInterval:
        averageScoresByInterval[inter] = sum(scoresByInterval[inter]) / len(scoresByInterval[inter])
    

    keyNames = ["0", "1", "2", "3", "4", "5", "6-10", "11-20", "> 20"]
    keyNames = [name for name in keyNames if name in list(averageScoresByInterval.keys())]
    print(keyNames)
    ks = list(range(len(keyNames)))

    values = [averageScoresByInterval[kn] for kn in keyNames]
    print(ks,values)

    plt.xticks(ks,keyNames)
    plt.xlabel("Numbers of decisions since last accept")
    plt.ylabel("average SVM confidence of accepted file")
    plt.bar(ks, values)
    plt.savefig(figname)
    plt.close()

def analyze_data(norm_anch, reviewer_decision, svm_decision, review_session):
    df = pd.DataFrame(norm_anch.cpu().detach().numpy(), columns = ['norm_anchor'])
    df['reviewer_decision'] = reviewer_decision.cpu()
    df['svm_decision'] = svm_decision.squeeze().cpu()
    admission = np.array(review_session)[:,2]
    df['admission'] = admission
    print(df)