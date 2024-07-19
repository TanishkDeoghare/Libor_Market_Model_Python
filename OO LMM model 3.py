import numpy as np

class SABR_LMM:
    def __init__(self, initial_rates, initial_volatilities, betas, correlation_matrix, alpha, rho, nu):
        self.initial_rates = initial_rates
        self.initial_volatilities = initial_volatilities
        self.betas = betas
        self.correlation_matrix = correlation_matrix
        self.alpha = alpha  # Volatility of volatility
        self.rho = rho  # Correlation between the forward rate and its volatility
        self.nu = nu  # Volatility of the volatility process
        self.num_tenors = len(initial_rates)
        self.chol_decomp = np.linalg.cholesky(correlation_matrix)
    
    def drift_term(self, L, sigma, t):
        # Implement the drift term using mean field theory approximation if needed
        return np.zeros(self.num_tenors)
    
    def diffusion_term(self, L, sigma, t):
        diffusion = sigma * L**self.betas
        return diffusion
    
    def simulate_step(self, L, sigma, dt, Z, Z_vol):
        drift = self.drift_term(L, sigma, t)
        diffusion = self.diffusion_term(L, sigma, t)
        dW = np.dot(self.chol_decomp, Z) * np.sqrt(dt)
        dZ = np.dot(self.chol_decomp, Z_vol) * np.sqrt(dt)
        
        L_new = L + drift * dt + diffusion * dW
        sigma_new = sigma * np.exp(self.alpha * dZ - 0.5 * self.alpha**2 * dt)
        
        return L_new, sigma_new

import matplotlib.pyplot as plt

class MonteCarloSimulation_SABR:
    def __init__(self, model, dt, total_time):
        self.model = model
        self.dt = dt
        self.total_time = total_time
        self.num_steps = int(total_time / dt)
    
    def simulate(self, num_paths):
        paths_L = np.zeros((num_paths, self.num_steps, self.model.num_tenors))
        paths_sigma = np.zeros((num_paths, self.num_steps, self.model.num_tenors))
        
        for i in range(num_paths):
            paths_L[i, 0] = self.model.initial_rates
            paths_sigma[i, 0] = self.model.initial_volatilities
            for t in range(1, self.num_steps):
                Z = np.random.normal(0, 1, self.model.num_tenors)
                Z_vol = np.random.normal(0, 1, self.model.num_tenors)
                paths_L[i, t], paths_sigma[i, t] = self.model.simulate_step(
                    paths_L[i, t-1], paths_sigma[i, t-1], self.dt, Z, Z_vol)
        return paths_L, paths_sigma
    
    def plot_paths(self, paths_L, paths_sigma, num_paths_to_plot=10):
        time_grid = np.linspace(0, self.total_time, self.num_steps)
        
        for j in range(self.model.num_tenors):
            plt.figure(figsize=(10, 6))
            for i in range(min(num_paths_to_plot, paths_L.shape[0])):
                plt.plot(time_grid, paths_L[i, :, j], label=f'Path {i+1}')
            plt.xlabel('Time')
            plt.ylabel(f'Forward Rate L{j+1}')
            plt.title(f'Simulated LIBOR Forward Rate L{j+1}')
            plt.legend()
            plt.show()
            
            plt.figure(figsize=(10, 6))
            for i in range(min(num_paths_to_plot, paths_sigma.shape[0])):
                plt.plot(time_grid, paths_sigma[i, :, j], label=f'Path {i+1}')
            plt.xlabel('Time')
            plt.ylabel(f'Volatility sigma{j+1}')
            plt.title(f'Simulated Volatility sigma{j+1}')
            plt.legend()
            plt.show()

from scipy.optimize import minimize

class Calibration_SABR:
    def __init__(self, model, market_forward_rates, market_volatilities):
        self.model = model
        self.market_forward_rates = market_forward_rates
        self.market_volatilities = market_volatilities
    
    def objective_function(self, params):
        self.model.alpha = params[:self.model.num_tenors]
        self.model.rho = params[self.model.num_tenors:2*self.model.num_tenors]
        self.model.nu = params[2*self.model.num_tenors:]
        
        simulated_paths_L, simulated_paths_sigma = MonteCarloSimulation_SABR(self.model, 0.01, 1.0).simulate(100)
        simulated_final_rates = np.mean(simulated_paths_L[:, -1, :], axis=0)
        simulated_final_volatilities = np.mean(simulated_paths_sigma[:, -1, :], axis=0)
        
        error_rates = np.sum((simulated_final_rates - self.market_forward_rates)**2)
        error_volatilities = np.sum((simulated_final_volatilities - self.market_volatilities)**2)
        return error_rates + error_volatilities
    
    def calibrate(self):
        initial_guess = np.concatenate([self.model.alpha, self.model.rho, self.model.nu])
        bounds = [(0.01, 1.0)] * self.model.num_tenors * 3
        result = minimize(self.objective_function, initial_guess, bounds=bounds)
        self.model.alpha = result.x[:self.model.num_tenors]
        self.model.rho = result.x[self.model.num_tenors:2*self.model.num_tenors]
        self.model.nu = result.x[2*self.model.num_tenors:]
        return result

import pandas as pd

# Load data from CSV
file_path = "C:/Users/tanishk.deoghare/OneDrive - Angel Oak Capital Advisors/Desktop/Libor Market Models/LiborMarketModels/Input Curves 3.xlsx"
data = pd.read_csv(file_path)

# Assuming the CSV has columns 'Maturity', 'InitialRates', and 'MarketForwardRates'
maturities = data['Maturity'].values
initial_rates = data['InitialRates'].values
market_forward_rates = data['MarketForwardRates'].values

# Example initial volatilities and parameters for SABR
initial_volatilities = np.random.uniform(0.1, 0.3, len(initial_rates))
betas = 0.5 * np.ones(len(initial_rates))
correlation_matrix = np.eye(len(initial_rates))
alpha = 0.3 * np.ones(len(initial_rates))
rho = -0.3 * np.ones(len(initial_rates))
nu = 0.4 * np.ones(len(initial_rates))

# Create the SABR-LMM model
sabr_lmm = SABR_LMM(initial_rates, initial_volatilities, betas, correlation_matrix, alpha, rho, nu)

# Calibrate the model using the market forward rates and market volatilities
market_volatilities = np.random.uniform(0.15, 0.25, len(initial_rates))  # Example market volatilities
calibration = Calibration_SABR(sabr_lmm, market_forward_rates, market_volatilities)
calibration_result = calibration.calibrate()

# Print calibrated parameters
print("Calibrated alpha:", sabr_lmm.alpha)
print("Calibrated rho:", sabr_lmm.rho)
print("Calibrated nu:", sabr_lmm.nu)

# Simulate paths with the calibrated model
simulation = MonteCarloSimulation_SABR(sabr_lmm, dt=0.01, total_time=1.0)
paths_L, paths_sigma = simulation.simulate(num_paths=1000)
simulation.plot_paths(paths_L, paths_sigma)
