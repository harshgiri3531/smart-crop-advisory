from flask import Flask, render_template, request
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from langchain_groq import ChatGroq   # ✅ CHANGED

app = Flask(__name__)

# ✅ Groq client
llm = ChatGroq(
    groq_api_key= os.getenv("GROQ_API_KEY") ,   
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

# Load Dataset
file_path ="C:\\Users\\gosai\\Downloads\\data for imageProcessing2.csv.xlsx"

df = pd.read_excel(file_path)

ndvi = df[9].dropna().reset_index(drop=True)

np.random.seed(42)
yield_data = (ndvi * 25) + np.random.normal(0, 1, len(ndvi))

data = pd.DataFrame({"NDVI": ndvi, "Yield": yield_data})

# Train ML Model
X = data[['NDVI']]
y = data['Yield']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor()
model.fit(X_train, y_train)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    crop_status = ""
    advice = ""
    yield_status = ""
    reply = ""
    user_msg = ""

    if request.method == "POST":

        # ✅ NDVI FORM (same as before)
        if "ndvi" in request.form:
            try:
                input_ndvi = float(request.form["ndvi"])
                user_input = pd.DataFrame({"NDVI": [input_ndvi]})
                prediction = round(model.predict(user_input)[0], 2)

                if input_ndvi > 0.6:
                    crop_status = (
                        "🟢 HEALTHY <br>"
                        " 👉 The crop is in good condition. Leaves have healthy green color "
                        "and growth is uniform across the field."
                    )

                    advice = (" <br> "
                        "➡️Maintain current irrigation schedule.<br>"
                        "➡️Keep monitoring the field once every 2–3 days.<br>"
                        "➡️Continue regular fertilizer plan without change.<br>"
                        "➡️Watch early signs of pest infestation during evening hours."
                    )

                elif input_ndvi > 0.3:
                    crop_status =( 
                        "🟡 MODERATE <br>"
                        " 👉 The crop is stable, but some parts of the field show early signs of stress."
                        " Timely care will help avoid losses."
                    )

                    advice = ( "<br>"
                        "➡️Check soil moisture and irrigate if top layer is dry.<br>"
                        "➡️Apply light dose of micronutrients (Zn/Fe) if leaf color is dull.<br>"
                        "➡️Inspect for early pest symptoms on lower leaves.<br>"
                        "➡️Remove weeds to reduce competition for nutrients and water.<br>"
                    )

                else:
                    crop_status =(
                        "🔴 STRESSED <br>"
                        " 👉 The crop is under high stress. Growth is weak, and plants are losing strength." 
                        "Immediate action is required to prevent yield reduction"
                    )

                    advice = ( " <br>"
                        "➡️Provide immediate irrigation (especially if NDVI is very low).<br>"
                        "➡️Check for pest/disease presence, treat if necessary.<br>"
                        "➡️Apply recommended NPK fertilizer to support plant recovery.<br>"
                        "➡️If patches are severely damaged, consider gap-filling.<br>"
                        "➡️Reduce waterlogging if present by opening drainage channels.<br>"
                    )

                if prediction < 15:
                    yield_status = "⚠ Expected Yield: LOW"
                elif prediction < 20:
                    yield_status = "⚠ Expected Yield: MEDIUM"
                else:
                    yield_status = "🌟 Expected Yield: HIGH"

            except:
                prediction = "Invalid input"

        # ✅ CHATBOT LOGIC (Groq version)
        elif "message" in request.form:
            user_msg = request.form["message"]
            language = "Hindi"
            try:
                prompt = f"""
You are an expert agriculture advisor for Indian farmers.
Instructions:
- Answer in {language}
- Give simple and practical advice
- Use point-wise format
- Each point in new line
Farmer Question: {user_msg}
"""

                response = llm.invoke(prompt)

                reply = response.content

            except Exception as e:
                reply = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        crop_status=crop_status,
        advice=advice,
        yield_status=yield_status,
        reply=reply,
        user_msg=user_msg
    )


@app.route("/knowmore")
def know_more():
    return render_template("knowmore.html")

@app.route("/data50")
def data50():
    table = df.head(50).to_html(classes="table table-striped", border=0)
    return render_template("data50.html", table=table)

@app.route("/images")
def show_images():
    images = [
        "Screenshot (13).png",
        "Screenshot (14).png",
        "Screenshot (15).png",
        "output.png",
    ]
    return render_template("images.html", images=images)


if __name__ == "__main__":
    app.run(debug=True)