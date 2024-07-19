import numpy as np

class LiborMarketModel:
    def __init__(self, initial_rates, volatilities, betas, correlation_matrix):
        self.initial_rates = initial_rates
        self.volatilities = volatilities
        self.betas = betas
        self.correlation_matrix = correlation_matrix
        self.num_tenors = len(initial_rates)
        self.chol_decomp = np.linalg.cholesky(correlation_matrix)
        
    def drift_term(self, L, t):
        # Placeholder for the drift term
        return np.zeros(self.num_tenors)
    
    def diffusion_term(self, L, t):
        diffusion = self.volatilities * L**self.betas
        return diffusion

    def simulate_step(self, L, dt, Z):
        drift = self.drift_term(L, 0)  # Assuming drift term is zero
        diffusion = self.diffusion_term(L, 0)
        dW = np.dot(self.chol_decomp, Z) * np.sqrt(dt)
        L_new = L + drift * dt + diffusion * dW
        return L_new

import matplotlib.pyplot as plt

class MonteCarloSimulation:
    def __init__(self, model, dt, total_time):
        self.model = model
        self.dt = dt
        self.total_time = total_time
        self.num_steps = int(total_time / dt)
    
    def simulate(self, num_paths):
        paths = np.zeros((num_paths, self.num_steps, self.model.num_tenors))
        for i in range(num_paths):
            paths[i, 0] = self.model.initial_rates
            for t in range(1, self.num_steps):
                Z = np.random.normal(0, 1, self.model.num_tenors)
                paths[i, t] = self.model.simulate_step(paths[i, t-1], self.dt, Z)
        return paths
    
    def plot_paths(self, paths, num_paths_to_plot=10):
        plt.figure(figsize=(10, 6))
        time_grid = np.linspace(0, self.total_time, self.num_steps)
        for i in range(min(num_paths_to_plot, paths.shape[0])):
            for j in range(self.model.num_tenors):
                plt.plot(time_grid, paths[i, :, j], label=f'Path {i+1}, L{j+1}')
        plt.xlabel('Time')
        plt.ylabel('Forward Rates')
        plt.title('Simulated LIBOR Forward Rates')
        plt.legend()
        plt.show()

from scipy.optimize import minimize

class Calibration:
    def __init__(self, model, market_data):
        self.model = model
        self.market_data = market_data
    
    def objective_function(self, params):
        self.model.volatilities = params[:self.model.num_tenors]
        self.model.betas = params[self.model.num_tenors:]
        # Perform simulation and calculate error with market data
        simulated_paths = MonteCarloSimulation(self.model, 0.01, 1.0).simulate(100)
        simulated_volatilities = np.std(simulated_paths[:, -1, :], axis=0)
        market_volatilities = self.market_data
        error = np.sum((simulated_volatilities - market_volatilities)**2)
        return error
    
    def calibrate(self):
        initial_guess = np.concatenate([self.model.volatilities, self.model.betas])
        bounds = [(0.01, 1.0)] * self.model.num_tenors + [(0, 1)] * self.model.num_tenors
        result = minimize(self.objective_function, initial_guess, bounds=bounds)
        self.model.volatilities = result.x[:self.model.num_tenors]
        self.model.betas = result.x[self.model.num_tenors:]
        return result

# Example initial rates, volatilities, betas, and correlation matrix
initial_rates = np.random.uniform(0.01, 0.05, 10)
volatilities = 0.2 * np.ones(10)
betas = 0.5 * np.ones(10)
correlation_matrix = np.eye(10)

# Create the model
lmm = LiborMarketModel(initial_rates, volatilities, betas, correlation_matrix)

# Simulate paths
simulation = MonteCarloSimulation(lmm, dt=0.01, total_time=1.0)
paths = simulation.simulate(num_paths=1000)
simulation.plot_paths(paths)

# Calibration example
market_volatilities = np.random.uniform(0.1, 0.3, 10)  # Example market data
calibration = Calibration(lmm, market_volatilities)
calibration_result = calibration.calibrate()

print("Calibrated volatilities:", lmm.volatilities)
print("Calibrated betas:", lmm.betas)
