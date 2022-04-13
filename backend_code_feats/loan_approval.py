import numpy as np
import pandas as pd
import pickle

# class for processing data in real-time and making predictions:
class LoanApprovalProcessing:
 
    # engineering and preprocessing numerical features:
    def engineer_new_feats(self, data):
        # engineering features in realtime:
        data['LogApplicantIncome'] = np.log(data['ApplicantIncome']+1)
        data['LogCoapplicantIncome'] = np.log(data['CoapplicantIncome']+1)
        data['LogLoanAmount'] = np.log(data['LoanAmount']+1)
        
        #   computing first time:
        data['CombinedNumFeat'] = (data['LogLoanAmount']*data['LogCoapplicantIncome'])/data['LogLoanAmount']
        
        #   final feats used in modelling:
        data['CombinedNumFeat'] = np.log(data['CombinedNumFeat']+1)
        data['Loan_Amount_Term'] = np.log(data['Loan_Amount_Term']+1)
        data['ApplicantIncome'] = data['LogApplicantIncome']
        return data

    # encoding categorical features in the form of one-hot encoding:
    def encoding_cat_feats(self, data, imp_cat_feats):
        feats_updated = []
        
        # encoding values for Credit_History:
        if data['Credit_History'] == 'Yes':
            data['Credit_History'] = 0.0
        else:
            data['Credit_History'] = 1.0
            feats_updated.append('Credit_History')
            
        # encoding values for Education:
        if data['Education'] == 'Graduate':
            data['Education'] = 0.0
        else:
            data['Education'] = 1.0
            feats_updated.append('Education')
            
        # encoding values for Married:
        if data['Married'] == 'Yes':
            data['Married'] = 0.0
        else:
            data['Married'] = 1.0
            feats_updated.append('Married')
            
        # encoding values for Self_Employed:
        if data['Self_Employed'] == 'Yes':
            data['Self_Employed'] = 1.0
            feats_updated.append('Self_Employed')
        else:
            data['Self_Employed'] = 0.0
            
        # encoding values for Dependents:
        if data['Dependents'] == 0:
            data['Dependents_0'] = 1.0
            feats_updated.append('Dependents_0')
        else:
            data['Dependents_0'] = 0.0
            
        # encoding values for property_area:
        if (data['Property_Area'] == 'Rural'):
            data['Property_Area_Rural'] = 1.0
            feats_updated.append('Property_Area_Rural')
            
        elif (data['Property_Area'] == 'Semiurban'):
            data['Property_Area_Semiurban'] = 1.0
            feats_updated.append('Property_Area_Semiurban')
            
        else:
            pass
        
        # non-updated feats:
        non_updated_feats = list(set(imp_cat_feats).symmetric_difference(set(feats_updated)))
        
        # assigning the non-updated feats to 0:  
        for feat in non_updated_feats:
            data[feat] = 0.0
            
        return data

    # bringing the data in array as it passed through the model: 
    def final_data_to_pass(self, data):
        final_data_pass = np.array([data['Credit_History'],
                                data['Education'],
                                data['Loan_Amount_Term'],
                                data['Married'],
                                data['Dependents_0'],
                                data['Property_Area_Semiurban'],
                                data['Self_Employed'],
                                data['CombinedNumFeat'],
                                data['Property_Area_Rural'],
                                data['ApplicantIncome']])
        return final_data_pass 

    # combining all of the above methods:
    def preprocess_at_one_go(self, data, imp_cat_feats):
        new_data = self.engineer_new_feats(data)
        cat_encoded_data = self.encoding_cat_feats(new_data, imp_cat_feats)
        final_data = self.final_data_to_pass(cat_encoded_data).reshape(1, 10)
        final_dataframe = pd.DataFrame(final_data, columns=['Credit_History', 'Education', 'Loan_Amount_Term', 'Married', 'Dependents_0', 'Property_Area_Semiurban', 'Self_Employed', 'CombinedNumFeat', 'Property_Area_Rural', 'ApplicantIncome'])
        print("Preprocessed Data --> ", final_data)
        return final_dataframe

    # loading model:
    def load_model(self):
        file_handle = open('best_voting_classifier_model.pkl', 'rb')
        voting_clf = pickle.load(file_handle)
        file_handle.close()
        return voting_clf
    
    # encoding final result:
    def encoding_result(self, result):
        if result['predicted_class'] == 0:
            final_message =  'Loan Should NOT be Approved. \n Probability of Not Approving Loan = {}'.format(result['class_0_prob'])
            class_ = result["predicted_class"]
        else:
            final_message =  'Loan SHOULD be Approved. \n Probability of Approving Loan = {}'.format(result['class_1_prob'])
            class_ = result["predicted_class"]
    
        return (final_message, class_) 

    # making final predictions using the Voting Classifier:
    def make_predictions(self, data_to_predict, imp_cat_feats):
        preprocessed_data = self.preprocess_at_one_go(data_to_predict, imp_cat_feats)
        voting_clf = self.load_model()
        pred_class = voting_clf.predict(preprocessed_data)[0]
        pred_class_prob = [prob for prob in voting_clf.predict_proba(preprocessed_data)[0]]
        result = {'predicted_class' : pred_class, 'class_0_prob': pred_class_prob[0], 'class_1_prob': pred_class_prob[1]}
        final_message, class_ = self.encoding_result(result)
        return (final_message, class_)

# final_result returned is of the form: [final_message, predicted_class]

# dummy data on which we are testing:
# dummy_data = {"Loan_Amount_Term": 120.0,
#        "ApplicantIncome": 79000.0,
#        "CoapplicantIncome": 1000.0,
#        "LoanAmount": 500.00,
#        "Credit_History": "Yes",
#        "Education": "Graduate",
#        "Married": "Yes",
#        "Self_Employed": "Yes",
#        "Dependents": 0,
#        "Property_Area": "Rural"}

# # Range of numerical feats values:
# # Range of ApplicantIncome --> 150 to 81000
# # Range of CoapplicantIncome --> 0.0 to 41667.0
# # Range of LoanAmount --> 9.0 to 700.0 
# # Values valid for Loan_Amount_Term --> [360., 120., 180.,  60., 300., 480., 240.,  36.,  84.,  12.]

# # important categorical feats which would be used while:
# imp_cat_feats = ['Credit_History', 'Education', 'Married', 'Dependents_0', 'Property_Area_Semiurban', 'Self_Employed', 'Property_Area_Rural']

# testing the above class:
# loan_def_proc_class = LoanApprovalProcessing()
# final_predictions = loan_def_proc_class.make_predictions(dummy_data, imp_cat_feats)
# print('Output --> ', final_predictions)
