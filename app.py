#from flask import Flask,render_template,request
from Source.ML_codes.logger import logging
from Source.ML_codes.exception import CustomException
import sys
from Source.ML_codes.components.data_ingestion import DataIngestion,DataIngestionConfig
#creates a flask application
#app=Flask(__name__)

'''@app.route("/",methods=["GET","POST"])
def form():
    if (request.method=="GET"):
        return render_template("form.html")
    else:
        squareFootage=request.form["squareFootage"]
        numBedroomse=request.form["numBedrooms"]
        numBathrooms=request.form["numBathrooms"]
        yearBuilt=request.form["yearBuilt"]
        lotSize=request.form["lotSize"]
        garageSize=request.form["garageSize"]
        neighborhoodQuality=request.form["neighborhoodQuality"]
'''


if __name__=="__main__":
    #app.run(debug=True)
    logging.info("The execution has started")

    try:
        #data_ingestion_config=DataIngestionConfig
        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)