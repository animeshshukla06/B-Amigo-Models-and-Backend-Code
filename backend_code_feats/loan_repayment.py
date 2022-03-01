import numpy as np
import joblib
import pandas as pd

class LoanRepaymentProcessing:

    # engineering new features based upon the data entered by the Bank Manager:
    def engineer_new_feats(self, data):
        # engineering features in realtime:
        data['int.rate'] = data['int.rate']/100
        data['log.int.rate'] = np.log(data['int.rate']+1)
        data['log.installment'] = np.log(data['installment']+1)
        data['poss.getting.loan'] = (data['inq.last.6mths'])/(data['dti']+1)
        data['loan.info'] = (data['installment']*data['int.rate'])/data['annual.inc']
        data['fico.to.delinq.2yrs'] = (data['fico']*data['annual.inc'])/(data['delinq.2yrs']+1)
        data['credit.value'] = data['days.with.cr.line']/(data['delinq.2yrs']+1)
        data['revol.bal.to.annual.inc'] = data['revol.bal']/data['annual.inc']
        return data

    # we also need to preprocess the data based upon Min-Max Scaling so that all the features 
    # have same scale:
    def scaling_num_data(self, data, imp_num_feats):
        min_val_feats = [3.3464, 612.0, 0.0, 0.0, 0.00010, 0.05826, 0.0, 1906560.00, 0.0, 203.357, 0.0]
        range_feats = [3.500, 215, 119, 11.25, 0.0314199, 0.12877, 29.96, 283602058.065, 33, 17437.60, 3.65463]
        for idx in range(len(imp_num_feats)):
            data[imp_num_feats[idx]] = (data[imp_num_feats[idx]] - min_val_feats[idx])/range_feats[idx]
        return data


    # preprocessing Categorical Features:
    def encoding_cat_feats(self, data, imp_cat_feats):
        feats_updated = []

        # assigning values for purchase related categorical feats:
        if (data['purpose'] == 'Credit Card'):
            data['purpose_credit_card'] = 1
            feats_updated.append('purpose_credit_card')
            
        elif (data['purpose'] == 'Educational'):
            data['purpose_educational'] = 1
            feats_updated.append('purpose_educational')
            
        elif (data['purpose'] == 'Major Purchase'):
            data['purpose_major_purchase'] = 1
            feats_updated.append('purpose_major_purchase')
            
        elif (data['purpose'] == 'Small Business'):
            data['purpose_small_business'] = 1
            feats_updated.append('purpose_small_business')
        else:
            pass

        # assigning values for pub.rec related categorical feats:
        if (data['pub.rec'] == 0):
            data['pub.rec_0'] = 1
            feats_updated.append('pub.rec_0')
            
        elif (data['pub.rec'] == 1):
            data['pub.rec_1'] = 1
            feats_updated.append('pub.rec_1')
            
        else:
            pass

        # assigning values for credit.policy:
        if (data['credit.policy'] == 'Yes'):
            data['credit.policy'] = 1
            feats_updated.append('credit.policy')
        else:
            data['credit.policy'] = 0
        
        # non-updated feats:
        non_updated_feats = list(set(imp_cat_feats).symmetric_difference(set(feats_updated)))
        
        # assigning the non-updated feats to 0:
        for feat in non_updated_feats:
            data[feat] = 0.0
            
        return data

    # bringing data to the format as we need to pass it through the trained model:
    def final_data_to_pass(self, data):
        final_data_pass = np.array([data['log.installment'],
                                data['fico'],
                                data['purpose_credit_card'],
                                data['revol.util'],
                                data['poss.getting.loan'],
                                data['loan.info'],
                                data['purpose_educational'],
                                data['credit.policy'],
                                data['log.int.rate'],
                                data['dti'],
                                data['fico.to.delinq.2yrs'],
                                data['pub.rec_0'],
                                data['purpose_major_purchase'],
                                data['inq.last.6mths'],
                                data['revol.bal.to.annual.inc'],
                                data['purpose_small_business'],
                                data['pub.rec_1'],
                                data['credit.value']])
        return final_data_pass

    # preprocessing data in one go:
    def preprocess_at_one_go(self, data, imp_cat_feats, imp_num_feats):
        new_data = self.engineer_new_feats(data)
        num_scaled_data = self.scaling_num_data(new_data, imp_num_feats)
        cat_scaled_data = self.encoding_cat_feats(num_scaled_data, imp_cat_feats)
        final_data = self.final_data_to_pass(cat_scaled_data).reshape(1, 18)
        final_data_rounded = [[round(data_point, 4) for data_point in final_data[0]]]
        print('Final Data -->', final_data_rounded)
        final_dataframe = pd.DataFrame(final_data_rounded, columns= ['log.installment', 'fico', 'purpose_credit_card', 'revol.util', 'poss.getting.loan', 'loan.info', 'purpose_educational','credit.policy', 'log.int.rate', 'dti', 'fico.to.delinq.2yrs',
       'pub.rec_0', 'purpose_major_purchase', 'inq.last.6mths','revol.bal.to.annual.income', 'purpose_small_business', 'pub.rec_1', 'credit.value'])

        # print(final_data)
        # print(final_dataframe)
        return final_dataframe

    # loading model:
    def load_model(self):
        joblib_svc_model = joblib.load("joblib_svc_lr_model.pkl")
        return joblib_svc_model

    # encoding final result:
    def encoding_result(self, result):
        final_result = []
        
        if result['predicted_class'] == 0:
            final_message =  'REPAYMENT might FAIL. \n Probability of failing to repay the amount= {} %'.format(result['class_0_prob']*100)
            class_ = result["predicted_class"]
            final_result.append(final_message)
            final_result.append(class_)

        else:
            final_message =  'REPAYMENT will be Successful. \n Probability of successfully repaying amount= {} %'.format(result['class_1_prob']*100)
            class_ = result["predicted_class"]
            final_result.append(final_message)
            final_result.append(class_)
            
        return final_result

    # making final predictions using the Voting Classifier:
    def make_predictions(self, data_to_predict, imp_cat_feats, imp_num_feats):
        preprocessed_data = self.preprocess_at_one_go(data_to_predict, imp_cat_feats, imp_num_feats)
        svc_clf = self.load_model()
        pred_class = svc_clf.predict(preprocessed_data)[0]
        pred_class_prob = [prob for prob in svc_clf.predict_proba(preprocessed_data)[0]]
        result = {'predicted_class' : pred_class, 'class_0_prob': pred_class_prob[0], 'class_1_prob': pred_class_prob[1]}
        final_result = self.encoding_result(result)
        return final_result

# dummy data on which prediction needs to be done:
# dummy data on which we are testing:
# test_data = {"installment": 194.71,
#         "fico": 692,
#         "revol.util": 94.2,
#         "annual.inc": 45667.134,
#         "int.rate": 10.39,
#         "dti": 7.5,
#         "inq.last.6mths": 1,
#         "days.with.cr.line": 1920,
#         "delinq.2yrs": 0,
#         "revol.bal": 4524,
#         "credit.policy": "Yes",
#         "purpose": "Debt Consolidation",
#        "pub.rec": 0}

# # # Range of Numerical Feats on which model is trained: 
# # # Range of installment --> 15.67 to 940.14
# # # Range of fico --> 612 to 827
# # # Range of revol.util --> 0.0 to 119.0
# # # Range of annual.inc --> 8846.482234853493 to 353790.1091275145
# # # Range of int.rate --> 0.06 to 0.2164
# # # Range of dti --> 0.0 to 29.96
# # # Range of inq.last.6mths --> 0 to 33
# # # Range of days.with.cr.line --> 178.95833330000002 to 17639.95833
# # # Range of delinq.2yrs --> 0 to 13
# # # Range of revol.bal --> 0 to 1207359

# # declaring imp_num_feats list:
# imp_num_feats = ['log.installment', 'fico', 'revol.util', 'poss.getting.loan', 'loan.info', 'log.int.rate', 'dti', 'fico.to.delinq.2yrs', 'inq.last.6mths', 'credit.value', 'revol.bal.to.annual.inc']
# imp_cat_feats = ['purpose_credit_card', 'purpose_educational', 'credit.policy', 'pub.rec_0', 'purpose_major_purchase', 'purpose_small_business', 'pub.rec_1']

# # creating the instance of LoanRepayemntProcessing:
# loan_repayment_proc = LoanRepaymentProcessing()
# final_result = loan_repayment_proc.make_predictions(data_to_predict=test_data, imp_cat_feats=imp_cat_feats, imp_num_feats=imp_num_feats)
# print(final_result)

# # columns=['log.installment', 'fico', 'purpose_credit_card', 'revol.util', 'poss.getting.loan', 'loan.info', 'purpose_educational', 'credit.policy', 'log.int.rate', 'dti', 'fico.to.delinq.2yrs',
# #                                         'pub.rec_0', 'purpose_major_purchase', 'inq.last.6mths', 'revol.bal.to.annual.inc', 'purpose_small_business', 'pub.rec_1', 'credit.value']
