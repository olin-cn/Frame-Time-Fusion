import numpy as np
import pandas as pd

A1, tau1 = 3.87821e-7, 0.34921
A2, tau2 = 3.50008e-7, 2.97588
A3, tau3 = 3.00247e-7, 30.47546
y0 = 4.46074e-8
TARGET_VOLTAGE = 3.3


I_max = A1 + A2 + A3 + y0
G = TARGET_VOLTAGE / I_max

def relaxation_current(t):
   
    return A1*np.exp(-t/tau1) + A2*np.exp(-t/tau2) + A3*np.exp(-t/tau3) + y0

def current_to_voltage(I):
  
    return I * G

def generate_sine_trajectory(width_pixels=100, height_pixels=50,
                             total_distance=40.0, amplitude=10.0,
                             wavelength=8.0, pixel_size=0.5, velocity=1.0,
                             sample_interval=0.1, noise_ratio=0.05):

    num_points = int(total_distance / velocity / sample_interval)
    x_phys = np.linspace(0, total_distance, num_points)
    y_phys = (height_pixels*pixel_size)/2 + amplitude * np.sin(2*np.pi*x_phys / wavelength)
    
   
    x_pix = np.clip(np.round(x_phys/pixel_size).astype(int), 0, width_pixels-1)
    y_pix = np.clip(np.round(y_phys/pixel_size).astype(int), 0, height_pixels-1)
    
 
    voltage_matrix = np.zeros((height_pixels, width_pixels))
    collection_time = total_distance / velocity 
    voltages = current_to_voltage(relaxation_current(collection_time - x_phys/velocity))
    
    for xi, yi, vi in zip(x_pix, y_pix, voltages):
        voltage_matrix[yi, xi] = max(voltage_matrix[yi, xi], vi)
    

    noise = np.random.normal(0, np.max(voltages)*noise_ratio, voltage_matrix.shape)
    voltage_matrix = np.maximum(0, voltage_matrix + noise)

    trajectory_points = []
    for xi, yi, vi, xp, yp in zip(x_pix, y_pix, voltages, x_phys, y_phys):
        trajectory_points.append({
            'x_pixel': int(xi),
            'y_pixel': int(yi),
            'x_meter': float(xp),
            'y_meter': float(yp),
            'voltage': float(vi)
        })
    
    return voltage_matrix, trajectory_points

def save_voltage_matrix(voltage_matrix, filename='sine_trajectory_100x50.csv'):

    np.savetxt(filename, voltage_matrix, delimiter=',', fmt='%.6f')
    return filename

def save_trajectory_points(trajectory_points, filename='sine_trajectory_points_100x50.csv'):

    df = pd.DataFrame(trajectory_points)
    df.to_csv(filename, index=False)
    return filename


if __name__ == "__main__":
    voltage_matrix, trajectory_points = generate_sine_trajectory()
    

    save_voltage_matrix(voltage_matrix)
    save_trajectory_points(trajectory_points)
    
    print("done！")
    print(f"VShape: {voltage_matrix.shape}")
    print(f"Points: {len(trajectory_points)}")