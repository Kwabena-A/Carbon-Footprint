import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from typing import List
app = FastAPI()

class Predictor():
    def __init__(self, args: list):
        self.model = joblib.load("emmissions_classifier.joblib")

        features = ['Body Type', 'Sex', 'Diet', 'How Often Shower', 'Heating Energy Source',
               'Transport', 'Vehicle Type', 'Social Activity', 'Monthly Grocery Bill',
               'Frequency of Traveling by Air', 'Vehicle Monthly Distance Km',
               'Waste Bag Size', 'Waste Bag Weekly Count', 'How Long TV PC Daily Hour',
               'How Many New Clothes Monthly', 'How Long Internet Daily Hour',
               'Energy efficiency', 'Recycling Score', 'Cooking With Score']

        self.df = {}

        for i in range(0, len(features)):
            self.df[features[i]] = [args[i]]

        self.df = pd.DataFrame(self.df)

        self.df['Vehicle Type'].replace(np.nan, 'none', inplace=True)
        self.df = self.df.dropna()

        self.df.reset_index(inplace=True)
        self.df = self.df.drop(["index"], axis=1)

        uniqueMapping = {
        'Body Type':['overweight', 'obese', 'underweight', 'normal'],

        'Sex':['female', 'male'],

        'Diet':['pescatarian', 'vegetarian', 'omnivore', 'vegan'],

        'How Often Shower':['daily', 'less frequently', 'more frequently', 'twice a day'],

        'Heating Energy Source':['coal', 'natural gas', 'wood', 'electricity'],

        'Transport':['public', 'walk/bicycle', 'private'],

        'Vehicle Type':['none', 'petrol', 'diesel', 'hybrid', 'lpg', 'electric'],

        'Social Activity':['often', 'never', 'sometimes'],

        'Frequency of Traveling by Air':['frequently', 'rarely', 'never', 'very frequently'],

        'Waste Bag Size':['large', 'extra large', 'small', 'medium'],

        'Energy efficiency':['No', 'Sometimes', 'Yes'],
        }

        for column in self.df.columns:
            try:
                intVal = int(self.df[column][0])
            except ValueError:
                if column not in ["Recycling Score", "Cooking With Score"]:
                    self.df[column][0] = uniqueMapping[column].index(self.df[column][0])


        recyclingDict = {
            "Metal" : 100,
            "Paper" : 70,
            "Plastic" : 50,
            "Glass" : 40
        }

        cookingWithDict = {
            "Microwave" : 10,
            "Airfryer" : 20,
            "Stove" : 30,
            "Oven" : 50,
            "Grill" : 120
        }

        def removeAll(myList, remove):
            newList = []
            for elementIndx in range(0, len(myList) - 1):
                if not ord(myList[elementIndx]) in remove:
                    newList.append(myList[elementIndx])
            return newList

        def convertLists(predictorObject: Predictor, column: str, newColumn: str, pointMap):

            predictorObject.df[newColumn] = np.nan
            for index in predictorObject.df.index:
                recycling = list(predictorObject.df[column][index])
                # elements are the ascii for: [ ' ] ,
                removeelements = [91, 39, 93, 44]
                recycling = removeAll(recycling, removeelements)
                recycling = "".join(recycling)
                recycling = recycling.split(" ")
                score = 0
                if recycling[0] != "":
                    for material in recycling:
                        score += pointMap[material]
                predictorObject.df.loc[index, newColumn] = score

            predictorObject.df = predictorObject.df.drop([column], axis = 1)

        convertLists(self, "Recycling Score", "Recycling Score New", recyclingDict)
        convertLists(self, "Cooking With Score", "Cooking With Score New", cookingWithDict)

        self.df.rename(columns={"Recycling Score New": "Recycling Score", "Cooking With Score New": "Cooking With Score"}, inplace=True)

        print(self.df)

    def getPrediction(self) -> int:
        return self.model.predict(self.df)

@app.get("/")
def read_root():
    return {"message" : "Welcome to the emission predictor API"}

@app.get("/make-prediction/")
def make_prediction(args: List[str] = Query(...)):
    predictor = Predictor(args)
    return {"args": args, "prediction": f"{predictor.getPrediction()}"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)






