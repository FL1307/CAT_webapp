from flask import Flask, jsonify, request,send_from_directory,session,request
import pandas as pd
import random
from scipy.optimize import minimize
import numpy as np
from copy import deepcopy


stop_threshold = 0.3
df = pd.read_excel('data_modified.xlsx', engine='openpyxl',converters={'Item Solution':str,
                                                            'Complete Item Code':str,
                                                            'a':float,
                                                            'b':float,
                                                            "Item":int,
                                                            "Set":int})

def index_to_item_id_wrapper(index_list):
    if isinstance(index_list, list):
        new_list = []
        for each in index_list:
            max_idx = int(df.iloc[[each]][['Item']].values[0][0])
            new_list.append(max_idx)

        return new_list
    else: 
        return int(df.iloc[[index_list]][['Item']].values[0][0])

def logistic_2pl(theta, a, b):
    a = np.array(a)
    theta = np.array(theta)
    b = np.array(b)
    return 1 / (1 + np.exp(-a * (theta - b)))

def choose_next_idx(theta, a, b, answered_list):
    p = logistic_2pl(theta, a, b)
    ones = np.ones_like(p)
    q = ones - p
    a_square = a * a
    info_list = a_square * p * q
    sem_array = deepcopy(info_list)
    mask = np.zeros_like(sem_array, dtype=bool)
    mask[answered_list] = True
    sem_array[~mask] = 0
    
    # Get the sum of the modified array A
    i_t_theta = np.sum(sem_array)
    standard_error = 1 / np.sqrt(i_t_theta).item()

    info_list[answered_list] = -10000

    max_idx = np.argmax(info_list).item()
    max_value = np.max(info_list).item()

    del sem_array
    return max_idx, standard_error, max_value



# Log-likelihood function for the 2PL model
def log_likelihood(theta, a, b, responses):
    probabilities = logistic_2pl(theta, a, b)
    ones = np.ones_like(probabilities)

    log_likelihood_value = np.sum(responses * np.log(probabilities) + (ones - responses) * np.log(ones - probabilities))
    return -log_likelihood_value  # Negative log-likelihood for minimization

# Estimate student's ability using MLE
def estimate_ability(a, b, responses, initial_theta):
    
    # Minimize the negative log-likelihood
    result = minimize(log_likelihood, initial_theta, args=(a, b, responses), method='L-BFGS-B',bounds=[(-5,5)],options={"maxiter":2})
    
    # Estimated ability
    estimated_theta = result.x[0]
    return estimated_theta

def retrieve_ab_by_id(q_id):
    return df.iloc[[q_id]][['a']].values[0][0],df.iloc[[q_id]][['b']].values[0][0]

def retrieve_solution_label_by_id(q_id):
    return df.iloc[[q_id]][['Item Solution']].values[0][0]

def retrieve_question_code_by_id(q_id):
    print("retrieve_question_code_by_id q_id",q_id)
    q_list = df.iloc[[q_id]][['Complete Item Code']].values[0][0].split(',')[:-1]
    q_list.append("00000000000000000000")

    return ",".join(q_list)

def get_avg_b_level():
    return df.loc[:, 'b'].mean()

def get_mid_idx():
    mean_b = df['b'].mean()

    # Compute the absolute difference between each value in column "b" and the mean
    df['diff_from_mean'] = (df['b'] - mean_b).abs()

    # Identify the index of the minimum difference
    closest_index = df['diff_from_mean'].idxmin()
    return int(closest_index)

app = Flask(__name__)
app.secret_key = 'BAD_SECRET_KEY'

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/matrices.css')
def css():
    return send_from_directory('.', 'matrices.css')

@app.route('/svg.js')
def svg_js():
    return send_from_directory('.', 'svg.js')

@app.route('/drawing.js')
def drawing_js():
    return send_from_directory('.', 'drawing.js')

@app.route('/api/data', methods=['POST','GET'])
def data():
    global examinee_responses
    global stop_threshold
    if request.method == 'POST':
        data = request.json
        # Process the data
        cur_response = data["response"]
        label = retrieve_solution_label_by_id(session["cur_q_id"])
        session["q_answered_list"].append(session["cur_q_id"])

        boolean = 0
        if cur_response == label:
            print("answer correct")
            boolean=1

        else: 
            print("answer incorrect")
            boolean=0



        cur_a, cur_b = retrieve_ab_by_id(session["cur_q_id"])
        session["a_list"].append(cur_a)
        session["b_list"].append(cur_b)
        session["res_list"].append(boolean)

        # Estimate ability
        ability_estimate = estimate_ability(session["a_list"],
                                            session["b_list"],
                                            session["res_list"],
                                            session["student_level"])
        session["student_level"] = ability_estimate.item()
        print("ability_estimate",ability_estimate.item())
        session["student_level_list"].append(round(session["student_level"], 3))
        
        all_a = df['a'].values
        all_b = df['b'].values

        new_id, se, max_value = choose_next_idx(session["student_level"], all_a, all_b, session["q_answered_list"])

        session["cur_q_id"] = new_id
        session["standard_error"].append(round(se, 3))
        session["information"].append(round(max_value, 3))
        print("current turn",session["cur_turn"])
        session["cur_turn"] += 1

        new_q = retrieve_question_code_by_id(session["cur_q_id"])
        
        print("current std error list",session["standard_error"])
        print("current information list",session["information"])
        print("current student_level_list",session["student_level_list"])
        print("current q_answered_list",session["q_answered_list"])
        if se < stop_threshold:
            summary = {
                "standard_error": session["standard_error"],
                "information": session["information"],
                "q_answered_list": index_to_item_id_wrapper(session["q_answered_list"]),
                "student_level_list": session["student_level_list"],
                "result_correctness_list":session["res_list"]
            }
            return jsonify({"summary": summary})

        return jsonify({'data':new_q})

    elif request.method == 'GET':
        # Initialize first question
        session["cur_q_id"]=get_mid_idx()
        session["a_list"] = []
        session["cur_turn"] = 1
        session["b_list"] = []
        session["res_list"] = []
        session["information"] = []
        session["q_answered_list"] = []
        session["standard_error"] = []
        session["student_level_list"] = []

        # Set initial student level to average b value.
        session["student_level"] = get_avg_b_level()
        print("initialize question idx =",session["cur_q_id"])
        print("initial student level",session["student_level"])
        
        return jsonify({'data':retrieve_question_code_by_id(session["cur_q_id"])})

@app.route('/api/turn', methods=['GET'])
def get_turn():
    return jsonify({"cur_turn": session["cur_turn"],\
                    "cur_id":index_to_item_id_wrapper(session["cur_q_id"])})

@app.route('/api/reset', methods=['POST'])
def reset_session():
    session.clear()
    return jsonify({"message": "Session reset"}), 200

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=80)