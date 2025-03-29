from flask import Flask,render_template,request

#creates a flask application
app=Flask(__name__)

@app.route("/",methods=["GET","POST"])
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



if __name__=="__main__":
    app.run(debug=True)