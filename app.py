from flask import Flask, request
from backend_code_feats.customer_churn import CustomerChurnProcessing # for customer churn preprocessing and predictions
from backend_code_feats.loan_approval import LoanApprovalProcessing # for loan approval preprocessing and predictions
from backend_code_feats.loan_repayment import LoanRepaymentProcessing # for loan repayement preprocessing and predictions
from backend_code_feats.fraudulent_transaction import DetectFraudulentTransaction # for detecting fraudulent transactions

# instance of Flask which helps in creating API'S:
app = Flask(__name__)

# example test router:
@app.route('/', methods=['GET', 'POST'])
def test_example():
    final_result = {"message": "API is working good", "status": "Success"}
    return final_result

# decorator function for customer churn prediction:
@app.route('/customer_churn', methods=['GET', 'POST'])
def customer_churn_prediction():
    
    # recieving the post data sent through the API in JSON format for which prediction needs to be done:
    data_dict = request.get_json()
    print(data_dict)
    
    # important numerical an categorical feats:
    imp_cat_feats = ['Gender', 'IsActiveMember', 'France', 'Germany', 'Spain']
    imp_num_feats = ['CreditScore', 'Age', 'Balance']
    
    # instance of class which preprocesse and passes data to the model:
    customer_churn_proc = CustomerChurnProcessing()
    final_message, class_ = customer_churn_proc.make_predictions(data_dict, imp_cat_feats, imp_num_feats)
    final_result = {"message": final_message, "class": class_}
    return final_result


# decorator function for customer Loan Approval/Defaulter Prediction:
@app.route('/loan_approval', methods=['GET', 'POST'])
def loan_approval_prediction():

    # recieving the post data sent through the API in JSON format for which prediction needs to be done:
    data_dict = request.get_json()
    print(data_dict)
    
    # important categorical feats which would be used while:
    imp_cat_feats = ['Credit_History', 'Education', 'Married', 'Dependents_0', 'Property_Area_Semiurban', 'Self_Employed', 'Property_Area_Rural']
    
    # instance of class which preprocesses and passes data to the model:
    loan_approval_proc = LoanApprovalProcessing()
    final_message, class_ = loan_approval_proc.make_predictions(data_dict, imp_cat_feats)
    final_result = {"message": final_message, "class": class_}
    return final_result


# decorator function for loan repayment prediction:
@app.route('/loan_repayment', methods=['GET', 'POST'])
def loan_repayment_prediction():

    # recieving the post data sent through the API in JSON format for which prediction needs to be done:
    data_dict = request.get_json()
    
    # important numerical and categorical feats:
    imp_num_feats = ['log.installment', 'fico', 'revol.util', 'poss.getting.loan', 'loan.info', 'log.int.rate', 'dti', 'fico.to.delinq.2yrs', 'inq.last.6mths', 'credit.value', 'revol.bal.to.annual.inc']
    imp_cat_feats = ['purpose_credit_card', 'purpose_educational', 'credit.policy', 'pub.rec_0', 'purpose_major_purchase', 'purpose_small_business', 'pub.rec_1']
    
    # instance of class which preprocesse and passes data to the model:
    loan_repayment_proc = LoanRepaymentProcessing()
    final_message, class_ = loan_repayment_proc.make_predictions(data_dict, imp_cat_feats, imp_num_feats)
    final_result = {"message": final_message, "class": class_}
    return final_result

# decorator function for detecting fraudulent transaction:
@app.route('/detect_fraud_transaction', methods=['GET', 'POST'])
def detect_fraud_transaction():

    # recieving the post data sent through the API in JSON format for which prediction needs to be done:
    data_dict = request.get_json()
    
    # important numerical and categorical feats:
    imp_num_feats = ['step', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'oldbalanceOrig', 'diff_orig']
    imp_cat_feats = ['diff_orig_dest', 'type_PAYMENT', 'type_TRANSFER', 'type_CASH_OUT', 'type_DEBIT', 'type_CASH_IN']

    # instance of class which preprocesse and passes data to the model:
    detect_fraud_transaction = DetectFraudulentTransaction()
    final_message, class_ = detect_fraud_transaction.make_predictions(data_dict, imp_cat_feats, imp_num_feats)
    final_result = {"message": final_message, "class": class_}
    return final_result


# running the file:
if __name__ == "__main__":
    app.run(debug=True)
