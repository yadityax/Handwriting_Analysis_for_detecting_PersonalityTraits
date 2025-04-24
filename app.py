from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
import os

from zones import findZone, zone_extract
from pressure import pressure, pressure_extract
from letterSize import letter_size_extract
from margin import margin_extract

from PIL import Image

import pickle

from rsvm import  predict_pressure, predict_zone, predict_margin, predict_letterSize, result_pressure, result_zone, result_letterSize, result_margin
import cv2

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "POST":
        listZone = ["Above", "Middle", "Below"]
        listPressure = ["Heavy", "Medium", "Light"]
        listBaseline = ["Falling", "Straight", "Rising"]
        listMargin = ["Narrow","Big"]
        listLetterSize = ["Small","Medium","Big"]
        if "test_image" in request.files:
            test_image = request.files["test_image"]
            if test_image.filename != "":
                # Save uploaded image file
                image_file = os.path.join("uploads", "test.png")
                test_image.save(image_file)

                # Extract zone information
                zone_info = zone_extract(image_file)
                pressure_info = pressure_extract(image_file)
                # baseline_info = baseline_extract(image_file)
                margin_info = margin_extract(image_file)
                letter_size_info = letter_size_extract(image_file)

                # Predict personality traits using zone information
                zone_result = predict_zone(np.array([zone_info]))
                pressure_result = predict_pressure(np.array([pressure_info]))
                # baseline_result = predict_baseline(np.array([baseline_info]))
                margin_result = predict_margin(np.array([margin_info]))
                letter_size_result = predict_letterSize(np.array([letter_size_info]))


                # Get personality descriptions
                zone_description = result_zone(zone_result[0])
                pressure_description = result_pressure(pressure_result[0])
                # baseline_description = result_baseline(baseline_result[0])
                margin_description = result_margin(margin_result[0])
                letter_size_description = result_letterSize(letter_size_result[0])

                # Convert int64 to regular integer
                zone_result = zone_result[0].item()
                pressure_result = pressure_result[0].item()
                # baseline_result = baseline_result[0].item()
                margin_result = margin_result[0].item()
                letter_size_result = letter_size_result[0].item()

                zone_class = listZone[zone_result-1]
                pressure_class = listPressure[pressure_result-1]
                # baseline_class = listBaseline[baseline_result-1]
                letter_size_class = listBaseline[letter_size_result-1]
                margin_class = listMargin[margin_result-1]
                letter_size_class = listLetterSize[margin_result-1]

                average = round(pressure_info[0],2)
                zone_above = zone_info[0]
                zone_middle = zone_info[1]
                zone_below = zone_info[2]
                
                margin = round(margin_info[0],2)
                letter_size = letter_size_info[0]
                

                description = {
                    "ZoneClass": zone_class,
                    "PressureClass": pressure_class,
                    "MarginClass": margin_class,
                    "LetterSizeClass": letter_size_class,
                    "ZoneDescription": zone_description,
                    "PressureDescription": pressure_description,
                    "MarginDescription": margin_description,
                    "LetterSizeDescription": letter_size_description,
                    "PressureInfo": average,
                    "Zone_Above" : zone_above,
                    "Zone_Middle" : zone_middle,
                    "Zone_Below" : zone_below,
                    "Margin": margin,
                    "LetterSize": letter_size
                }
                return render_template("test.html", description=description)
    # Return a response for GET requests or if the conditions are not met
    return render_template("test.html")

if __name__ == "__main__":
    app.run(debug=True)
