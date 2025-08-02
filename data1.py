import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

csv_data = """Date;Location;CarsPassed;Energy_kWh;PeakTime;CO2_Saved_kg;LightsPowered_hr;BatteryLevel_%
Jul 1, 2025;Al Mahalah District;282;30.91;8–10PM;7.73;12.4;100%
Jul 1, 2025;Muftaha Area;221;21.48;7–9PM;5.37;8.6;75%
Jul 1, 2025;Al Hizam Road;269;22.03;9–11PM;5.51;8.8;77%
Jul 2, 2025;Al Soudah Entrance;220;21.72;8–10PM;5.43;8.7;76%
Jul 2, 2025;Abha Dam Park;184;15.16;8–10PM;3.79;6.1;53%
Jul 2, 2025;Airport Road;221;18.21;9–11PM;4.55;7.3;64%
Jul 2, 2025;Abha Mall Parking;214;22.46;9–11PM;5.62;9;79%
Jul 3, 2025;Tuesday Market Zone;264;28.98;8–10PM;7.25;11.6;100%
Jul 3, 2025;Al Soudah Entrance;294;29.4;9–11PM;7.35;11.8;100%
Jul 3, 2025;Al Hizam Road;260;25.74;9–11PM;6.43;10.3;90%
Jul 3, 2025;Al Raqa Neighborhood;223;24.33;9–11PM;6.08;9.7;85%
Jul 4, 2025;Al Raqa Neighborhood;173;14.97;8–10PM;3.74;6;52%
Jul 4, 2025;Al Raqa Neighborhood;290;24.91;9–11PM;6.23;10;87%
Jul 4, 2025;Al Raqa Neighborhood;292;33.2;7–9PM;8.3;13.3;100%
Jul 4, 2025;Airport Road;175;17.54;8–10PM;4.38;7;61%
Jul 4, 2025;Al Raqa Neighborhood;190;22.1;7–9PM;5.53;8.8;77%
Jul 4, 2025;Al Hizam Road;233;26.67;9–11PM;6.67;10.7;93%
Jul 5, 2025;Al Mi'ah Street;162;15.59;8–10PM;3.9;6.2;55%
Jul 5, 2025;Al Hizam Road;215;25.6;9–11PM;6.4;10.2;90%
Jul 5, 2025;Airport Road;159;15.08;8–10PM;3.77;6;53%
Jul 6, 2025;King Khalid University Gate;211;24.14;8–10PM;6.04;9.7;84%
Jul 6, 2025;Al Hizam Road;249;22.75;8–10PM;5.69;9.1;80%
Jul 6, 2025;Al Mahalah District;249;28.83;7–9PM;7.21;11.5;100%
Jul 6, 2025;Abha Dam Park;211;24.2;8–10PM;6.05;9.7;85%
Jul 7, 2025;Al Hizam Road;234;23.02;8–10PM;5.75;9.2;81%
Jul 7, 2025;Al Mahalah District;295;25.14;7–9PM;6.29;10.1;88%
Jul 7, 2025;Al Soudah Entrance;197;17.35;8–10PM;4.34;6.9;61%
Jul 8, 2025;Abha Dam Park;141;12.51;8–10PM;3.13;5;44%
Jul 8, 2025;Abha Mall Parking;248;23.59;8–10PM;5.9;9.4;83%
Jul 8, 2025;Al Soudah Entrance;216;23.92;9–11PM;5.98;9.6;84%
Jul 8, 2025;Muftaha Area;246;29.3;9–11PM;7.33;11.7;100%
Jul 8, 2025;Airport Road;260;28.18;7–9PM;7.04;11.3;99%
Jul 9, 2025;Airport Road;228;18.64;7–9PM;4.66;7.5;65%
Jul 9, 2025;Abha Mall Parking;264;27.87;9–11PM;6.97;11.1;98%
Jul 9, 2025;Tuesday Market Zone;146;12.28;8–10PM;3.07;4.9;43%
Jul 9, 2025;Al Raqa Neighborhood;195;18.1;8–10PM;4.53;7.2;63%
Jul 10, 2025;Abha Dam Park;288;25.33;8–10PM;6.33;10.1;89%
Jul 10, 2025;Al Raqa Neighborhood;249;20.76;9–11PM;5.19;8.3;73%
Jul 10, 2025;Abha Mall Parking;244;26.23;7–9PM;6.56;10.5;92%
Jul 10, 2025;Al Hizam Road;232;19.62;8–10PM;4.91;7.8;69%
Jul 10, 2025;King Khalid University Gate;207;24.19;9–11PM;6.05;9.7;85%
Jul 10, 2025;Tuesday Market Zone;253;29.29;9–11PM;7.32;11.7;100%
Jul 11, 2025;Tuesday Market Zone;295;30.5;7–9PM;7.62;12.2;100%
Jul 11, 2025;Al Hizam Road;175;20.09;8–10PM;5.02;8;70%
Jul 11, 2025;Al Mi'ah Street;153;15.95;9–11PM;3.99;6.4;56%
Jul 11, 2025;Abha Mall Parking;227;25.53;9–11PM;6.38;10.2;89%
Jul 11, 2025;Muftaha Area;290;28.03;8–10PM;7.01;11.2;98%
Jul 11, 2025;Al Mahalah District;209;17.43;8–10PM;4.36;7;61%
Jul 12, 2025;Al Raqa Neighborhood;241;22.64;9–11PM;5.66;9.1;79%
Jul 12, 2025;Al Soudah Entrance;263;22.04;8–10PM;5.51;8.8;77%
Jul 12, 2025;Al Hizam Road;144;11.89;7–9PM;2.97;4.8;42%
Jul 13, 2025;Abha Dam Park;189;21.1;9–11PM;5.28;8.4;74%
Jul 13, 2025;Al Soudah Entrance;219;25.27;9–11PM;6.32;10.1;88%
Jul 13, 2025;Al Hizam Road;306;31.72;9–11PM;7.93;12.7;100%
Jul 13, 2025;Airport Road;196;17.25;7–9PM;4.31;6.9;60%
Jul 13, 2025;Al Mahalah District;254;22;9–11PM;5.5;8.8;77%
Jul 13, 2025;Tuesday Market Zone;244;22.5;7–9PM;5.62;9;79%
Jul 14, 2025;Al Soudah Entrance;278;26.86;9–11PM;6.71;10.7;94%
Jul 14, 2025;Al Raqa Neighborhood;234;27.65;8–10PM;6.91;11.1;97%
Jul 14, 2025;King Khalid University Gate;264;22.75;7–9PM;5.69;9.1;80%
Jul 14, 2025;Abha Mall Parking;266;25.69;8–10PM;6.42;10.3;90%
Jul 14, 2025;Airport Road;281;24.65;9–11PM;6.16;9.9;86%
Jul 15, 2025;Abha Dam Park;245;26.26;9–11PM;6.57;10.5;92%
Jul 15, 2025;Abha Mall Parking;293;27.07;8–10PM;6.77;10.8;95%
Jul 15, 2025;Al Mi'ah Street;206;18.92;7–9PM;4.73;7.6;66%
Jul 15, 2025;Al Soudah Entrance;212;24.65;7–9PM;6.16;9.9;86%
Jul 15, 2025;Al Raqa Neighborhood;221;25.01;8–10PM;6.25;10;88%
Jul 15, 2025;Al Mahalah District;336;39.76;9–11PM;9.94;15.9;100%
Jul 16, 2025;Airport Road;179;15.87;9–11PM;3.97;6.3;56%
Jul 16, 2025;Al Hizam Road;217;20.78;8–10PM;5.2;8.3;73%
Jul 16, 2025;Al Soudah Entrance;240;23.87;9–11PM;5.97;9.5;84%
Jul 16, 2025;Abha Mall Parking;244;21.7;9–11PM;5.42;8.7;76%
Jul 16, 2025;Tuesday Market Zone;184;17.47;9–11PM;4.37;7;61%
Jul 16, 2025;Al Mi'ah Street;222;21.93;8–10PM;5.48;8.8;77%
Jul 17, 2025;King Khalid University Gate;190;16.07;9–11PM;4.02;6.4;56%
Jul 17, 2025;Airport Road;265;27.86;7–9PM;6.96;11.1;98%
Jul 17, 2025;Al Mi'ah Street;276;31.4;7–9PM;7.85;12.6;100%
Jul 17, 2025;Al Hizam Road;201;17.22;8–10PM;4.3;6.9;60%
Jul 17, 2025;Abha Mall Parking;196;22;8–10PM;5.5;8.8;77%
Jul 18, 2025;Abha Dam Park;257;27.48;9–11PM;6.87;11;96%
Jul 18, 2025;Al Raqa Neighborhood;272;22.17;7–9PM;5.54;8.9;78%
Jul 18, 2025;Al Hizam Road;287;23.73;9–11PM;5.93;9.5;83%
Jul 18, 2025;Al Mi'ah Street;256;26.22;9–11PM;6.55;10.5;92%
Jul 18, 2025;Tuesday Market Zone;202;23.93;9–11PM;5.98;9.6;84%
Jul 18, 2025;King Khalid University Gate;257;27.58;8–10PM;6.89;11;97%
Jul 19, 2025;Al Soudah Entrance;229;19.64;8–10PM;4.91;7.9;69%
Jul 19, 2025;Airport Road;170;19.59;9–11PM;4.9;7.8;69%
Jul 19, 2025;King Khalid University Gate;213;19.61;8–10PM;4.9;7.8;69%
Jul 19, 2025;Abha Dam Park;238;20.29;9–11PM;5.07;8.1;71%
Jul 19, 2025;Al Mi'ah Street;238;27.73;7–9PM;6.93;11.1;97%
Jul 19, 2025;Muftaha Area;315;34.02;7–9PM;8.51;13.6;100%
Jul 20, 2025;Al Mi'ah Street;244;21.21;8–10PM;5.3;8.5;74%
Jul 20, 2025;Al Hizam Road;194;18.76;8–10PM;4.69;7.5;66%
Jul 20, 2025;Al Raqa Neighborhood;204;21.81;9–11PM;5.45;8.7;76%
Jul 20, 2025;Abha Mall Parking;239;20.89;8–10PM;5.22;8.4;73%
Jul 21, 2025;Al Mahalah District;173;17.67;9–11PM;4.42;7.1;62%
Jul 21, 2025;Al Mi'ah Street;204;23.87;7–9PM;5.97;9.5;84%
Jul 21, 2025;Al Hizam Road;195;22.52;9–11PM;5.63;9;79%
Jul 21, 2025;Abha Dam Park;233;26.38;7–9PM;6.59;10.6;92%
Jul 21, 2025;Muftaha Area;297;25.82;7–9PM;6.46;10.3;90%
Jul 22, 2025;Al Raqa Neighborhood;157;15.93;7–9PM;3.98;6.4;56%
Jul 22, 2025;Al Soudah Entrance;327;36.33;8–10PM;9.08;14.5;100%
Jul 22, 2025;King Khalid University Gate;328;35.02;7–9PM;8.76;14;100%
Jul 22, 2025;Abha Dam Park;163;17.49;7–9PM;4.37;7;61%
Jul 22, 2025;Abha Mall Parking;314;27.93;8–10PM;6.98;11.2;98%
Jul 22, 2025;Al Mahalah District;311;33.01;7–9PM;8.25;13.2;100%
Jul 23, 2025;King Khalid University Gate;269;30.38;9–11PM;7.59;12.2;100%
Jul 23, 2025;Al Raqa Neighborhood;280;22.88;9–11PM;5.72;9.2;80%
Jul 23, 2025;Al Hizam Road;310;26.51;7–9PM;6.63;10.6;93%
Jul 23, 2025;Abha Dam Park;203;20.26;7–9PM;5.07;8.1;71%
Jul 24, 2025;Tuesday Market Zone;262;24.24;7–9PM;6.06;9.7;85%
Jul 24, 2025;King Khalid University Gate;181;17.25;9–11PM;4.31;6.9;60%
Jul 24, 2025;Al Hizam Road;236;27.22;7–9PM;6.8;10.9;95%
Jul 25, 2025;Al Raqa Neighborhood;220;22.45;8–10PM;5.61;9;79%
Jul 25, 2025;Muftaha Area;269;28.39;9–11PM;7.1;11.4;99%
Jul 25, 2025;Al Mi'ah Street;290;31.59;9–11PM;7.9;12.6;100%
Jul 26, 2025;Al Raqa Neighborhood;178;19.38;7–9PM;4.84;7.8;68%
Jul 26, 2025;Airport Road;260;26.46;8–10PM;6.62;10.6;93%
Jul 26, 2025;Al Hizam Road;140;15.64;7–9PM;3.91;6.3;55%
Jul 26, 2025;Al Mahalah District;242;22.71;9–11PM;5.68;9.1;79%
Jul 26, 2025;Al Soudah Entrance;271;22.26;9–11PM;5.57;8.9;78%
Jul 27, 2025;Abha Dam Park;176;17.86;7–9PM;4.46;7.1;63%
Jul 27, 2025;Abha Mall Parking;290;26.11;7–9PM;6.53;10.4;91%
Jul 27, 2025;Tuesday Market Zone;259;27.7;7–9PM;6.92;11.1;97%
"""

df = pd.read_csv(io.StringIO(csv_data), sep=';')

print("Data after reading:")
print(df.head())
print("\n" + "-"*50 + "\n")

df['BatteryLevel_%'] = df['BatteryLevel_%'].str.replace('%', '', regex=False).astype(float)
average_energy = df['Energy_kWh'].mean()
df['High_Energy_Day'] = (df['Energy_kWh'] > average_energy).astype(int)

features = df[['CarsPassed', 'Energy_kWh', 'CO2_Saved_kg', 'BatteryLevel_%']]
target = df['High_Energy_Day']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

today_data = pd.DataFrame({
    'CarsPassed': [250],
    'Energy_kWh': [25.0],
    'CO2_Saved_kg': [6.0],
    'BatteryLevel_%': [90.0]
})
today_prediction = model.predict(today_data)

print("\n" + "-"*50 + "\n")
print(f"The average energy consumption in the data is: {average_energy:.2f} kWh")
print(f"New day's data for prediction: \n{today_data}")
print("\n" + "-"*50 + "\n")

if today_prediction[0] == 1:
    print("Prediction: It is a high energy day (above average).")
else:
    print("Prediction: It is a normal energy day (below average).")