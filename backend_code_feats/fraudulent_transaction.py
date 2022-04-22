import numpy as np
import joblib
import pandas as pd

class DetectFraudulentTransaction:

    # engineering new features based upon the data entered by the Bank Manager:
    def engineer_new_feats(self, data):
        # engineering features in realtime:
        data["diff_orig"] = data["newbalanceOrig"] - data["oldbalanceOrig"]
        data["diff_dest"] = data["newbalanceDest"] - data["oldbalanceDest"]
        
        if (data["diff_orig"]==data["diff_dest"]):
            data["diff_orig_dest"] = 1.0
        else:
            data["diff_orig_dest"] = 0.0
            
        return data

    # we also need to preprocess the data based upon Min-Max Scaling so that all the features 
    # have same scale:
    def scaling_num_data(self, data, imp_num_feats):
        min_val_feats = [1.0, 0.0, 0.0, 0.0, 0.0, -10000000.00]
        range_feats = [9.0, 38946233.02, 34008736.98, 38946233.02, 38939424.03, 11289407.91]
        for idx in range(len(imp_num_feats)):
            data[imp_num_feats[idx]] = (data[imp_num_feats[idx]] - min_val_feats[idx])/range_feats[idx]
        return data


    # preprocessing Categorical Features:
    def encoding_cat_feats(self, data, imp_cat_feats):
        feats_updated = []

        # assigning values for purchase related categorical feats:
        if (data['type'] == 'PAYMENT'):
            data['type_PAYMENT'] = 1
            feats_updated.append('type_PAYMENT')
            
        elif (data['type'] == 'TRANSFER'):
            data['type_TRANSFER'] = 1
            feats_updated.append('type_TRANSFER')
            
        elif (data['type'] == 'CASH_OUT'):
            data['type_CASH_OUT'] = 1
            feats_updated.append('type_CASH_OUT')
            
        elif (data['type'] == 'DEBIT'):
            data['type_DEBIT'] = 1
            feats_updated.append('type_DEBIT')
            
        elif (data['type'] == 'CASH_IN'):
            data['type_CASH_IN'] = 1
            feats_updated.append('type_CASH_IN')
            
        else:
            pass
        
        # non-updated feats:
        non_updated_feats = list(set(imp_cat_feats).symmetric_difference(set(feats_updated)))
        
        # assigning the non-updated feats to 0:
        for feat in non_updated_feats:
            data[feat] = 0.0
            
        return data

    # bringing data to the format as we need to pass it through the trained model:
    def final_data_to_pass(self, data):
        final_data_pass = np.array([data['step'],
                                data['newbalanceOrig'],
                                data['oldbalanceDest'],
                                data['newbalanceDest'],
                                data['oldbalanceOrig'],
                                data['diff_orig'],
                                data['diff_orig_dest'],
                                data['type_PAYMENT'],
                                data['type_TRANSFER'],
                                data['type_CASH_OUT'],
                                data['type_DEBIT'],
                                data['type_CASH_IN']])
        return final_data_pass

    # preprocessing data in one go:
    def preprocess_at_one_go(self, data, imp_cat_feats, imp_num_feats):
        new_data = self.engineer_new_feats(data)
        num_scaled_data = self.scaling_num_data(new_data, imp_num_feats)
        cat_scaled_data = self.encoding_cat_feats(num_scaled_data, imp_cat_feats)
        final_data = self.final_data_to_pass(cat_scaled_data).reshape(1, 12)
        final_dataframe = pd.DataFrame(final_data, columns = ['step', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
       'oldbalanceOrig', 'diff_orig', 'diff_orig_dest', 'type_PAYMENT',
       'type_TRANSFER', 'type_CASH_OUT', 'type_DEBIT', 'type_CASH_IN'])
        # print(final_data)
        print(final_dataframe)
        print("Columns --> {}".format(final_dataframe.columns))
        return final_dataframe

    # loading model:
    def load_model(self):
        fd_model = joblib.load("joblib_local_outlier_factor_fd.pkl")
        return fd_model

    # encoding final result:
    def encoding_result(self, result):
        
        if (result['predicted_class'] == -1):
            final_message = 'Fraudulent Transaction'
            class_ = 0.0
        else:
            final_message = 'Normal Transaction'
            class_ = result["predicted_class"]
            
        return (final_message, class_)

    # making final predictions using the Voting Classifier:
    def make_predictions(self, data_to_predict, imp_cat_feats, imp_num_feats):
        preprocessed_data = self.preprocess_at_one_go(data_to_predict, imp_cat_feats, imp_num_feats)
        model = self.load_model()
        pred_class = model.predict(preprocessed_data)[0]
        result = {'predicted_class' : pred_class}
        final_message, class_ = self.encoding_result(result)
        return (final_message, float(class_))


# testing the above deployement code:

# dummy data on which prediction needs to be done:
# dummy data on which we are testing:
# test_data = {"step": 3.0,
#            "newbalanceOrig": 2121.2,
#            "oldbalanceDest": 2233.2,
#            "newbalanceDest": 34590.2,
#            "oldbalanceOrig": 233343.98,
#           "type": "TRANSFER"}

# # # Range of Numerical Feats on which model is trained: 
# # # Range of step --> 1 to 10
# # # Range of newbalanceOrig --> 0.0 to 38946233.02
# # # Range of oldbalanceDest --> 0 to 34008736.98
# # # Range of newbalanceDest --> 0 to 38946233.02
# # # Range of oldbalanceOrig --> 0 to 38939424.03


# declaring imp_num_feats list:
# imp_num_feats = ['step', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'oldbalanceOrig', 'diff_orig']
# imp_cat_feats = ['diff_orig_dest', 'type_PAYMENT', 'type_TRANSFER', 'type_CASH_OUT', 'type_DEBIT', 'type_CASH_IN']

# #  creating the instance of DetectFraudulentTransaction :
# det_fraud_transaction = DetectFraudulentTransaction()
# final_result = det_fraud_transaction.make_predictions(data_to_predict=test_data, imp_cat_feats=imp_cat_feats, imp_num_feats=imp_num_feats)
# print(final_result)
