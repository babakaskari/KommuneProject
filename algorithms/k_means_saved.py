import pickle
import numpy as np



def k_means_model():
    with open('./k_means_model.pkl', 'rb') as file:
        pickle_model = pickle.load(file)
    var = [23.24102771, 5.319835068]
    arr = np.array(var)
    print("actual data : ", arr)
    arr = np.reshape(arr, (1, -1))
    prediction = pickle_model.predict(arr)
    if prediction[0] == 1:
        print("Thers is a leak alarm")
    else:
        print("There is no leak alaram")