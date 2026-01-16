from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from scipy.signal import detrend

app = Flask(__name__)

def compute_theoretical_baseline(L, b, h, E, rho, support):
    I = (b * h**3) / 12
    A = b * h
    mu = rho * A

    if support == "cantilever":
        betaL = 1.87510407
    elif support == "simply_supported":
        betaL = np.pi
    elif support == "fixed_fixed":
        betaL = 4.73004074
    else:
        betaL = 1.87510407

    beta2 = betaL ** 2
    omega = beta2 * np.sqrt(E * I / (mu * L**4))
    return omega / (2 * np.pi)

def analyze_csv(file, baseline):
    df = pd.read_csv(file)

    if not {"Accel_X", "Accel_Y", "Accel_Z"}.issubset(df.columns):
        return {"error": "CSV must contain: Accel_X, Accel_Y, Accel_Z"}

    ax = detrend(df["Accel_Z"].values)

    fs = 50.0
    freqs = np.fft.rfftfreq(len(ax), 1 / fs)
    mag = np.abs(np.fft.rfft(ax))

    dominant = freqs[np.argmax(mag)]
    diff = abs(dominant - baseline) / baseline * 100

    if diff < 20:
        level = "Healthy"
    elif diff < 50:
        level = "Minor Damage"
    elif diff < 80:
        level = "Moderate Damage"
    else:
        level = "Major Damage"

    return {
        "dominant": float(dominant),
        "baseline": float(baseline),
        "diff": float(diff),
        "level": level
    }

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        try:
            L = float(request.form["L"])
            b = float(request.form["b"])
            h = float(request.form["h"])
            E = float(request.form["E"]) * 1e9
            rho = float(request.form["rho"])
            support = request.form["support"]

            baseline = compute_theoretical_baseline(L, b, h, E, rho, support)

            file = request.files.get("csvfile")
            if not file:
                result = {"error": "Please upload a CSV file!"}
            else:
                result = analyze_csv(file, baseline)

        except Exception as e:
            result = {"error": str(e)}

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)