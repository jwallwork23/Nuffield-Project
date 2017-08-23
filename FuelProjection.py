# Import required packages:
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# List countries, initial populations and growth rates:
country = {1: 'Cuba', 2: 'Japan', 3: 'South Africa', 4: 'Uganda'}
P0 = {1: 11.4e6, 2: 127e6, 3: 53.3e6, 4: 37.5e6}
P_rate = {1: 1.39e-3, 2: 9.35e-3, 3: 1.57e-2, 4: 3.31e-2}

# List initial fuel values and consumption rates:
B0 = {1: 6.08e8, 2: 4.9e9, 3: 1.81e9, 4: 4.61e8}
C0 = {1: 2.44e7, 2: 5.23e8, 3: 30e9, 4: 3.49e5}
G0 = {1: 1.71e11, 2: 3.51e11, 3: 3.17e7, 4: 2.41e10}
O0 = {1: 2.43e7, 2: 3.69e8, 3: 4.53e10, 4: 4.13e8}
B_rate = {1: 1.39e-3, 2: 1.84e-3, 3: 1.89e-3, 4: 2.70}
C_rate = {1: 1.91e-3, 2: 1.50, 3: 2.95, 4: 0.}
G_rate = {1: 0.122, 2: 0.014, 3: 0.123, 4: 0.}
O_rate = {1: 0.11, 2: 1.54, 3: 0.72, 4: 2.09e-3}
B_CO2 = 'To do'
C_C02 = 2.30e-3
G_CO2 = 3.36e-8
O_CO2 = 3.32e-3

# Create dictionaries to store data:
P = {1: [], 2: [], 3: [], 4: []}
B = {1: [], 2: [], 3: [], 4: []}
C = {1: [], 2: [], 3: [], 4: []}
G = {1: [], 2: [], 3: [], 4: []}
O = {1: [], 2: [], 3: [], 4: []}

# Carrying capacity:
K = {1: 1.52e7, 2: 1.69e8, 3: 7.11e7, 4: 5e7}

# Other parameters:
g = float(raw_input('Biomass growth rate? (Default 0.01): ') or 0.01)
W = 7.5e9       # World population

# Choose population model and simulation duration:
model = raw_input('Choose model: Malthus or Logistic? (Default Malthus): ') or 'Malthus'
T = int(raw_input('Simulation duration in years? (Default 100): ') or 100)

# Plotting setup:
styles = {1: ':', 2: '--', 3: '-.', 4: '-'}
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('legend', fontsize='x-large')
plt.clf()
t_axis = np.linspace(0, T, T * 10)

# Plot population curves:
for i in country:
    if model == 'Malthus':
        for t in t_axis:
            P[i].append(P0[i] * np.exp(P_rate[i] * t))
        plt.semilogy(t_axis, P[i], label=country[i], linestyle=styles[i])
    elif model == 'Logistic':
        for t in t_axis:
            P[i].append((P0[i] * K[i] * np.exp(P_rate[i] * t)) / (K[i] + P0[i] * (np.exp(P_rate[i] * t) - 1)))
        plt.semilogy(t_axis, P[i], label=country[i], linestyle=styles[i])
    else:
        raise ValueError('Model selection not recognised.')
plt.gcf()
plt.legend(loc=2)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Population')
plt.savefig('plots/population_' + model + '.pdf', bbox_inches='tight')

# Plot coal curves:
plt.clf()
for i in country:
    if model == 'Malthus':
        for t in t_axis:
            C[i].append(C0[i] - C_rate[i] * P0[i] * (np.exp(P_rate[i] * t) - 1) / P_rate[i])
        plt.semilogy(t_axis, C[i], label=country[i], linestyle=styles[i])
    else:
        raise NotImplementedError('Model not yet considered.')
plt.gcf()
plt.legend(loc=4)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Coal (tonnes)')
plt.savefig('plots/coal_' + model + '.pdf', bbox_inches='tight')

# Plot gas curves:
plt.clf()
for i in country:
    if model == 'Malthus':
        for t in t_axis:
            G[i].append(G0[i] - G_rate[i] * P0[i] * (np.exp(P_rate[i] * t) - 1) / P_rate[i])
        plt.semilogy(t_axis, G[i], label=country[i], linestyle=styles[i])
    else:
        raise NotImplementedError('Model not yet considered.')
plt.gcf()
plt.legend(loc=4)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Gas (tonnes)')
plt.savefig('plots/gas_' + model + '.pdf', bbox_inches='tight')

# Plot oil curves:
plt.clf()
for i in country:
    if model == 'Malthus':
        for t in t_axis:
            O[i].append(O0[i] - O_rate[i] * P0[i] * (np.exp(P_rate[i] * t) - 1) / P_rate[i])
        plt.semilogy(t_axis, O[i], label=country[i], linestyle=styles[i])
    else:
        raise NotImplementedError('Model not yet considered.')
plt.gcf()
plt.legend(loc=4)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Oil (tonnes)')
plt.savefig('plots/oil_' + model + '.pdf', bbox_inches='tight')

# Plot biofuel curves:
plt.clf()
for i in country:
    if model == 'Malthus':
        for t in t_axis:
            B[i].append(B0[i] * np.exp(g * t)
                     + B_rate[i] * P0[i] * (np.exp(P_rate[i] * t) - np.exp(g * t)) / (g - P_rate[i]))
        plt.semilogy(t_axis, B[i], label=country[i], linestyle=styles[i])
    else:
        raise NotImplementedError('Model not yet considered.')
plt.gcf()
plt.legend(loc=4)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Biomass (tonnes)')
plt.savefig('plots/bio_' + model + '.pdf', bbox_inches='tight')