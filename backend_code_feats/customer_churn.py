import numpy as np
import pickle
import pandas as pd

class CustomerChurnProcessing:
    
    # scaling numerical features based upon Min-Max Scaling:
    def scaling_num_data(self, data, imp_num_feats):
        min_val_feats = [381.0, 18.0, 0.0]
        range_feats = [469.0, 46.0, 250898.09]

        for idx in range(len(imp_num_feats)):
            data[imp_num_feats[idx]] = (data[imp_num_feats[idx]] - min_val_feats[idx])/range_feats[idx]
        return data

    # processing and encoding categorical features:
    def encoding_cat_feats(self, data, imp_cat_feats):
        features_updated = []
            
        # assigning values of Gender:
        if (data['Gender'] == 'Male'):
            data['Gender'] = 1.0
            features_updated.append('Gender')
        else:
            data['Gender'] = 0.0
        
        # assigning values of isActiveMember:
        if (data['IsActiveMember'] == 'Yes'):
            data['IsActiveMember'] = 1.0
            features_updated.append('IsActiveMember')
        else:
            data['IsActiveMember'] = 0.0
            
        # assigning values of Geography:
        if (data['Geography'] == 'France'):
            data['France'] = 1.0
            features_updated.append('France')
        elif (data['Geography'] == 'Spain'):
            data['Spain'] = 1.0
            features_updated.append('Spain')
        elif (data['Geography'] == 'Germany'):
            data['Germany'] = 1.0
            features_updated.append('Germany')
        else:
            pass
        
        # non-updated feats:
        features_not_upd = list(set(imp_cat_feats).symmetric_difference(set(features_updated)))
        
        # assigning the non-updated feats to 0:
        for not_upd_feat in features_not_upd:
            data[not_upd_feat] = 0.0

        return data

    # bringing data in a form so as to pass it through the model nicely:
    def final_data_to_pass(self, data):
        final_data_pass = np.array([data['CreditScore'],
                              data['Gender'],
                              data['Age'],
                              data['Balance'],
                              data['NumOfProducts'],
                              data['IsActiveMember'],
                              data['France'],
                              data['Germany'],
                              data['Spain']])
        return final_data_pass

    # performing all of the above steps in a single method:
    def preprocess_at_one_go(self, dummy_data, imp_cat_feats, imp_num_feats):
        num_scaled_data = self.scaling_num_data(dummy_data, imp_num_feats)
        cat_scaled_data = self.encoding_cat_feats(num_scaled_data, imp_cat_feats)
        final_data = self.final_data_to_pass(cat_scaled_data).reshape(1, 9)
        feats_name = ['CreditScore', 'Gender', 'Age', 'Balance', 'NumOfProducts','IsActiveMember', 'France', 'Germany', 'Spain']
        final_dataframe = pd.DataFrame(final_data, columns=feats_name)
        print('Final Preprocessed Data -->', final_data)
        # print('Final Data Frame -->', final_dataframe)
        return final_dataframe

    # loading model:
    def load_model(self):
        file_handle = open('svc_model_cc.pkl', 'rb')
        svc_clf = pickle.load(file_handle)
        file_handle.close()
        return svc_clf

    # encoding final result:
    def encoding_result(self, result):
        if result['predicted_class'] == 0:
            final_message = 'Customer would STAY. \n Probability of customer staying= {} %'.format(result['class_0_prob']*100)
            class_ = result["predicted_class"]
        else:
            final_message =  'Customer would LEAVE. \n Probability of customer leaving= {} %'.format(result['class_1_prob']*100)
            class_ = result["predicted_class"]

        return (final_message, class_)

    # making final predictions using the Voting Classifier:
    def make_predictions(self, data_to_predict, imp_cat_feats, imp_num_feats):
        preprocessed_data = self.preprocess_at_one_go(dummy_data = data_to_predict, imp_cat_feats=imp_cat_feats, imp_num_feats=imp_num_feats)
        svc_clf = self.load_model()
        pred_class = svc_clf.predict(preprocessed_data)[0]
        pred_class_prob = [prob for prob in svc_clf.predict_proba(preprocessed_data)[0]]
        result = {'predicted_class' : pred_class, 'class_0_prob': pred_class_prob[0], 'class_1_prob': pred_class_prob[1]}
        final_message, class_ = self.encoding_result(result)
        return (final_message, class_)

# dummy_data for testing:
# dummy_data = {"CreditScore": 450, 
#             "Gender": "Male", 
#             "Age": 19,  
#             "Balance": 0, 
#             "NumOfProducts": 3, 
#             "IsActiveMember": "No", 
#             "Geography": "France"}

# # Range of Numerical features:
# # Range of CreditScore --> 381.0 to 850.0
# # Range of Age --> 18.0 to 64.0
# # Range of Balance --> 0.0 to 250898.09

# important numerical an categorical feats:
# imp_cat_feats = ['Gender', 'IsActiveMember', 'France', 'Germany', 'Spain']
# imp_num_feats = ['CreditScore', 'Age', 'Balance']

# # instance of CustomerChurnProcessing and making predictions:
# customer_churn_proc = CustomerChurnProcessing()
# final_result = customer_churn_proc.make_predictions(dummy_data, imp_cat_feats, imp_num_feats)
# print(final_result)
