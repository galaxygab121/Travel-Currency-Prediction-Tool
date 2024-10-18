import requests  # For making API requests
import pandas as pd  # For handling data manipulation
from tkinter import *  # For creating the GUI
from tkinter import ttk, messagebox  # Additional Tkinter modules for comboboxes and popups
from tkcalendar import Calendar  # A calendar widget for selecting travel dates
from datetime import datetime, timedelta  # For handling dates and time deltas
from keras.models import Sequential  # Sequential model from Keras for building the LSTM
from keras.layers import LSTM, Dense  # LSTM and Dense layers for neural network
import numpy as np  # For array and numerical calculations
import matplotlib.pyplot as plt  # For plotting data (if needed)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # For embedding matplotlib plots into the GUI

# Define the API key and base URL for the currency exchange API
api_key = 'f7a784a711c74cbad8952712'
base_url = 'https://v6.exchangerate-api.com/v6'

# Fetch the real-time exchange rate between two currencies
def fetch_real_time_exchange_rate(from_currency, to_currency):
    try:
        # Construct the URL with the API key and the "from" currency
        url = f'{base_url}/{api_key}/latest/{from_currency}'
        response = requests.get(url)  # Send the request to the API

        # If the request fails, raise an exception
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

        data = response.json()  # Parse the JSON response
        conversion_rate = data['conversion_rates'].get(to_currency)  # Get the conversion rate for the target currency
        
        # If no rate is found, raise a value error
        if conversion_rate is None:
            raise ValueError(f"No conversion rate found for {to_currency}.")
        
        return conversion_rate  # Return the conversion rate
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch exchange rate: {str(e)}")
        return None  # Return None if there's an error

# Fetch historical data to train the LSTM model (currently using dummy data)
def fetch_historical_data(base_currency, target_currency):
    # Simulating historical data for the last 60 days
    historical_data = np.array([1 + i * 0.01 for i in range(60)])  # Dummy data
    return historical_data  # Return the dummy data (replace with real historical data)

# Build and compile an LSTM model using Keras
def build_lstm_model():
    model = Sequential()  # Initialize a sequential model
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))  # Add an LSTM layer with 50 units and return sequences
    model.add(LSTM(50, return_sequences=False))  # Another LSTM layer without returning sequences
    model.add(Dense(25))  # Add a dense layer with 25 neurons
    model.add(Dense(1))  # Output layer with one neuron
    model.compile(optimizer='adam', loss='mean_squared_error')  # Compile the model with Adam optimizer and mean squared error loss
    return model  # Return the compiled model

# Predict future exchange rates using the trained LSTM model
def predict_exchange_rate_lstm(base_currency, target_currency, num_days):
    historical_data = fetch_historical_data(base_currency, target_currency)  # Fetch historical data
    
    model = build_lstm_model()  # Build the LSTM model
    
    # Reshape the historical data for the LSTM model
    X_test = np.reshape(historical_data, (1, 60, 1))  # Reshape the data into 3D for LSTM input
    
    # Generate predictions for the number of days the user has specified
    predictions = []
    for i in range(num_days):
        predicted_rate = model.predict(X_test)[0][0]  # Make the prediction
        predictions.append(predicted_rate)  # Append the predicted rate to the list
        # Update the historical data with the new prediction for sliding window effect
        historical_data = np.append(historical_data[1:], predicted_rate)
        X_test = np.reshape(historical_data, (1, 60, 1))  # Reshape again for the next prediction
    
    return predictions  # Return the predicted exchange rates

# Main GUI Application
class TravelCurrencyApp(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)  # Initialize the Tkinter root window
        self.title("Travel Currency Prediction Tool")  # Set window title
        self.geometry("1000x800")  # Set the size of the window
        
        # Display the title label
        Label(self, text="Travel Currency Prediction Tool", font=("Arial", 18)).pack(pady=20)
        
        # Country Selection Comboboxes (Home and Destination)
        Label(self, text="Home Country", font=("Arial", 12)).pack(pady=5)
        self.home_country = ttk.Combobox(self, values=['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD'])
        self.home_country.pack()
        
        Label(self, text="Destination Country", font=("Arial", 12)).pack(pady=5)
        self.destination_country = ttk.Combobox(self, values=['JPY', 'USD', 'EUR', 'GBP', 'CAD'])
        self.destination_country.pack()

        # Calendar for selecting travel start date
        Label(self, text="Travel Start Date", font=("Arial", 12)).pack(pady=5)
        self.calendar = Calendar(self, selectmode="day")
        self.calendar.pack(pady=5)

        # Input for travel duration in days
        Label(self, text="Travel Duration (days)", font=("Arial", 12)).pack(pady=5)
        self.duration = Entry(self)  # Input field for duration
        self.duration.pack()

        # Button to predict the best travel dates based on exchange rate prediction
        Button(self, text="Predict Best Travel Dates", command=self.predict_best_dates, font=("Arial", 12)).pack(pady=20)

        # Frame to hold the results (predictions for travel dates)
        self.results_frame = Frame(self)
        self.results_frame.pack(pady=10)

    # Function to predict best travel dates
    def predict_best_dates(self):
        # Fetch user inputs (currencies and duration)
        from_currency = self.home_country.get()
        to_currency = self.destination_country.get()
        start_date = self.calendar.get_date()
        duration = int(self.duration.get())  # Convert the duration to an integer

        if not from_currency or not to_currency or not start_date or not duration:
            messagebox.showerror("Error", "Please fill in all fields.")  # Error if inputs are missing
            return

        # Fetch the real-time exchange rate for the selected currencies
        real_time_rate = fetch_real_time_exchange_rate(from_currency, to_currency)
        if real_time_rate is None:
            return  # If there is an error, return early

        # Predict future exchange rates for the specified travel duration
        predicted_rates = predict_exchange_rate_lstm(from_currency, to_currency, duration)

        # Clear the previous results from the results frame
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        # Start date of the travel converted to a datetime object
        travel_start = datetime.strptime(start_date, "%m/%d/%y")
        
        # Iterate through each day of the travel duration
        for day in range(duration):
            travel_day = travel_start + timedelta(days=day)  # Calculate the travel day
            rate = predicted_rates[day]  # Get the predicted rate for that day
            
            # Color-code based on the prediction
            if rate < real_time_rate * 0.98:
                color = 'green'  # Favorable day
                label_text = f"Day {travel_day.strftime('%Y-%m-%d')} (Favorable): Rate = {rate:.4f}"
            elif real_time_rate * 0.98 <= rate <= real_time_rate * 1.02:
                color = 'yellow'  # Least favorable day
                label_text = f"Day {travel_day.strftime('%Y-%m-%d')} (Least Favorable): Rate = {rate:.4f}"
            else:
                color = 'red'  # Non-favorable day
                label_text = f"Day {travel_day.strftime('%Y-%m-%d')} (Not Favorable): Rate = {rate:.4f}"

            # Display the prediction as clickable label in the results frame
            label = Label(self.results_frame, text=label_text, fg=color, cursor="hand2")
            label.pack()
            label.bind("<Button-1>", lambda e, rate=rate: self.show_detailed_prediction(rate))  # Bind click event to show details

    # Display detailed prediction info in a message box
    def show_detailed_prediction(self, rate):
        messagebox.showinfo("Detailed Prediction", f"Predicted Exchange Rate: {rate:.4f}")

# Run the GUI application
app = TravelCurrencyApp()
app.mainloop()
