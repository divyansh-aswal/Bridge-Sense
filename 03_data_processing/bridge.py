import numpy as np
import pandas as pd
from scipy.signal import detrend

def compute_theoretical_baseline(L, b, h, E, rho, support):
    I = (b * h**3) / 12
    A = b * h
    mu = rho * A

    # first-mode βL constants
    if support == "cantilever":
        betaL = 1.87510407
    elif support == "simply_supported":
        betaL = np.pi
    elif support == "fixed_fixed":
        betaL = 4.73004074
    else:
        betaL = 1.87510407

    beta2 = betaL ** 2

    # bending frequency formula
    omega = beta2 * np.sqrt(E * I / (mu * L**4))
    return omega / (2 * np.pi)

def analyze_from_data(baseline):
    df = pd.read_csv("data.csv")

    if not {"Accel_X", "Accel_Y", "Accel_Z"}.issubset(df.columns):
        print("ERROR: CSV must contain: Accel_X, Accel_Y, Accel_Z")
        return

    ax = detrend(df["Accel_Z"].values)

    fs = 50.0
    freqs = np.fft.rfftfreq(len(ax), 1 / fs)
    mag = np.abs(np.fft.rfft(ax))

    dominant = freqs[np.argmax(mag)]
    diff = abs(dominant - baseline) / baseline * 100

    # damage classification
    if diff < 20:
        level = "Healthy"
    elif diff < 50:
        level = "Minor Damage"
    elif diff < 80:
        level = "Moderate Damage"
    else:
        level = "Major Damage"

    print("\n===========================")
    print(f"Dominant Frequency:   {dominant:.3f} Hz")
    print(f"Theoretical Baseline: {baseline:.3f} Hz")
    print(f"Percent Difference:   {diff:.2f}%")
    print(f"Damage Level:         {level}")
    print("===========================\n")

def main():
    print("\nENTER BRIDGE PARAMETERS:\n")

    L = float(input("Span Length L (m): "))
    b = float(input("Width b (m): "))
    h = float(input("Depth h (m): "))
    E = float(input("Elastic Modulus E (GPa): ")) * 1e9
    rho = float(input("Density ρ (kg/m³): "))
    support = input("Support type [simply_supported/cantilever/fixed_fixed]: ")

    baseline = compute_theoretical_baseline(L, b, h, E, rho, support)
    print(f"\nTheoretical Baseline Frequency = {baseline:.4f} Hz")

    analyze_from_data(baseline)

if __name__ == "__main__":
    main()

