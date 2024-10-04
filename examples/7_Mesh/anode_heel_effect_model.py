# ************************************************************************
# * This file is part of GGEMS.                                          *
# *                                                                      *
# * GGEMS is free software: you can redistribute it and/or modify        *
# * it under the terms of the GNU General Public License as published by *
# * the Free Software Foundation, either version 3 of the License, or    *
# * (at your option) any later version.                                  *
# *                                                                      *
# * GGEMS is distributed in the hope that it will be useful,             *
# * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
# * GNU General Public License for more details.                         *
# *                                                                      *
# * You should have received a copy of the GNU General Public License    *
# * along with GGEMS.  If not, see <https://www.gnu.org/licenses/>.      *
# *                                                                      *
# ************************************************************************

import numpy as np
import pandas as pd
import pathlib
import joblib

# Function to generate phi_weights and phi_angles based on kvp (energy)
def get_phi_distribution(kvp):
    # Load the scaler and model
    current_path = pathlib.Path(__file__).parent.resolve()
    scaler = joblib.load(current_path / "./data/scaler.pkl")
    gbr_model = joblib.load(current_path / "./data/gbr_model.pkl")

    # Distances from -9 to 9 (inclusive)
    distances = np.arange(-9, 10, 1)

    # Calculate Phis using the formula: theta = atan(distance / 70)
    phi_angles = np.arctan(distances / 70)

    # Use the mean DoseRate from your dataset as a placeholder
    mean_doserate = 0.004181104805128205  # This would be dataset specific

    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'Distance': distances,
        'DoseRate': mean_doserate,  # Mean doserate used here
        'Energy': kvp,  # Using kvp as the energy value
        'Theta': phi_angles,
        'ThetaDegrees': np.degrees(phi_angles)  # Convert radians to degrees
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict the weights
    phi_weights = gbr_model.predict(input_data_scaled)

    # Adjust angles (shift them by 90 degrees in radians)
    for i in range(len(phi_angles)):
        phi_angles[i] = phi_angles[i] + 90 * np.pi / 180

    phi_weights[0] = 0

    return phi_weights, phi_angles

