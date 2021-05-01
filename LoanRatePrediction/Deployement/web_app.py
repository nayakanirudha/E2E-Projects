from flask import Flask, render_template,request
from flasgger import Swagger
import pickle
import pandas as pd

app = Flask(__name__)
test_df = pd.read_csv("test_dataset.csv")
pickle_file = open('logisticRegression.pkl','rb')
classifier = pickle.load(pickle_file)
Swagger(app)

@app.route("/")
def base_route():
    return "Welcome to Loan Rate Prediction API",200

@app.route("/predictForSample",methods=['GET'])
def predictRate():
    """Swagger App for Loan Rate Prediction
    --------
    parameters:
    -   name: amount
        description: Listed Amount of the loan applied for by the borrower
        in: query
        type: integer
        required: true


    -   name: length_employed
        description : Employment length in years
        in: query
        type: integer
        required: true

    -   name: home_owner
        description : Home Ownership status provided by the borrower
        in: query
        type: integer
        required: true

    -   name: income
        description : Annual Income provided by the borrower
        in: query
        type: float
        required: true

    -   name: income_verified
        description : If income was verified, not verified, or if income source was verified
        in: query
        type: integer
        required: true

    -   name: purpose
        description : Loan Category provided by the borrower
        in: query
        type: integer
        required: true

    -   name: d2i
        description : Ratio of Borrowers Monthly debt payments to Monthly income
        in: query
        type: float
        required: true

    -   name: inquiries
        description : Inquiries by Creditors during last 6 month
        in: query
        type: integer
        required: true

    -   name: open_accounts
        description : Open Credit Lines in the borrowers credit file
        in: query
        type: integer
        required: true

    -   name: total_accounts
        description : Total Credit Lines in the borrowers credit file
        in: query
        type: integer
        required: true

    -   name: invalid_accounts
        description : Closed Credit Lines in the borrowers credit file
        in: query
        type: integer
        required: true

    -   name: gender
        description : Gender
        in: query
        type: integer
        required: true

    -   name: years2repay
        description : Total Loan Amount divided by Annual Income
        in: query
        type: float
        required: true

    -   name: income_label1
        description : Low Income
        in: query
        type: integer
        required: true

    -   name: income_label2
        description : Medium Income
        in: query
        type: integer
        required: true

    -   name: income_label3
        description : High Income
        in: query
        type: integer
        required: true

    -   name: loan_label1
        description : Small Loan
        in: query
        type: integer
        required: true

    -   name: loan_label2
        description : Medium Loan
        in: query
        type: integer
        required: true

    -   name: loan_label3
        description : High Loan
        in: query
        type: integer
        required: true

    -   name: deliquency
        description : History of Default
        in: query
        type: integer
        required: true

    responses:
        200:
            description : Predicted for Sample Customers
        201:
            description : Predicted for file containing all Customers

    """

    amount = request.args.get("Loan_Request_Amount")
    length_employed = request.args.get("Length_Employed")
    home_owner = request.args.get("Home_Owner")
    income = request.args.get("Annual_Income")
    income_verified = request.args.get("Income_Verified")
    purpose = request.args.get("Purpose_Of_Loan")
    d2i = request.args.get("Debt_To_Income")
    inquiries = request.args.get("Inquiries_Last_6Mo")
    open_accounts = request.args.get("Number_Open_Accounts")
    total_accounts = request.args.get("Total_Accounts")
    invalid_accounts = request.args.get("Number_Invalid_Acc")
    gender = request.args.get("Gender")
    years2repay = request.args.get("Number_Years_To_Repay_Debt")
    income_label1 = request.args.get("Income_label_1")
    income_label2 = request.args.get("Income_label_2")
    income_label3 = request.args.get("Income_label_3")
    loan_label1 = request.args.get("Loan_label_1")
    loan_label2 = request.args.get("Loan_label_2")
    loan_label3 = request.args.get("Loan_label_3")
    deliquency = request.args.get("Deliquency")

    result = classifier.predict([[amount, length_employed, home_owner, income,
                                  income_verified,purpose,d2i,inquiries,open_accounts,
                                  total_accounts,gender,invalid_accounts,years2repay,
                                  deliquency,income_label1,income_label2,income_label3,
                                  loan_label1,loan_label2,loan_label3]])

    if(result in [1.0,"1.0"]) : return "Loan Rate Category: 1"
    if(result in [2.0,"2.0"]) : return "Loan Rate Category: 2"
    if(result in [3.0,"3.0"]) : return "Loan Rate Category: 3"

@app.route("/predictForLoanID",methods=['GET'])
def predictRateforCustomer():
    """Swagger App for Loan Rate Prediction
    --------
    -   name: Loan_ID
        description : Enter the Loan ID of applicants 10164310 - 10273850
        in: query
        type: integer
        required: true

    responses:
        200:
            description : Predicted for file containing all Customers
    """
    print("Enter Loan ID of Applicants (Number between 10164310 - 10273850)")
    Loan_ID = request.args.get("Loan_ID")
    applicant_data = test_df[test_df["Loan_ID"] == int(Loan_ID)]
    result = classifier.predict(applicant_data.drop(["Loan_ID", "Interest_Rate"], axis=1).values)

    if(result in [1.0,"1.0"]) : return "Predicted Loan Rate Category: 1"
    if(result in [2.0,"2.0"]) : return "Predicted Loan Rate Category: 2"
    if(result in [3.0,"3.0"]) : return "Predicted Loan Rate Category: 3"

if __name__ == "__main__":
    app.run(debug=True, host= "127.0.0.1", port= 5000)

