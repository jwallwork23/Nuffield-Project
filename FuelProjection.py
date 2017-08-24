# Import required packages:
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def biomass_quadrature_step(P0, K, r, g, t, h):
    """
    A quadrature routine for approximating the increase in integral involved in the biomass logistic model solution.
    
    :param P0: initial human population.
    :param K: carrying capacity of region considered.
    :param r: rate of growth of population.
    :param g: biomass growth rate.
    :param t: end of time period considered.
    :param h: strip width.
    :return: integral in time interval [t-1, t].
    """
    I = 0.
    for j in range(int(t / h) - 1, int(t / h)):
        term = np.exp((r - g) * j * h) / (K + P0 * (np.exp(r * j * h) - 1))
        if j in (0, int(t / h) -1):
            I += term / 2
        else:
            I += term
    return I * h


# List countries, initial populations and growth rates:
country = {1: 'Cuba', 2: 'Japan', 3: 'South Africa', 4: 'Uganda'}
P0 = {1: 11.4e6, 2: 127e6, 3: 53.3e6, 4: 37.5e6}
P_rate = {1: 1.39e-3, 2: 9.35e-3, 3: 1.57e-2, 4: 3.31e-2}
WP0 = 7.5e9                                                 # World population

# List initial fuel values and consumption rates:       TODO: combine world values as if they were a country
B0 = {1: 6.08e8, 2: 4.9e9, 3: 1.81e9, 4: 4.61e8}
C0 = {1: 2.44e7, 2: 5.23e8, 3: 30e9, 4: 3.49e5}
G0 = {1: 1.71e11, 2: 3.51e11, 3: 3.17e7, 4: 2.41e10}
O0 = {1: 2.43e7, 2: 3.69e8, 3: 4.53e10, 4: 4.13e8}
WB0 = 2.4e13
WC0 = 4.7e12
WG0 = 8.7e12
WO0 = 1.9e12
B_rate = {1: 1.39e-3, 2: 1.84e-3, 3: 1.89e-3, 4: 2.70}
C_rate = {1: 1.91e-3, 2: 1.50, 3: 2.95, 4: 0.}
G_rate = {1: 0.122, 2: 0.014, 3: 0.123, 4: 0.}
O_rate = {1: 0.11, 2: 1.54, 3: 0.72, 4: 2.09e-3}
B_CO2 = 5.4e-2
C_CO2 = 2.30e-3
G_CO2 = 3.36e-8
O_CO2 = 3.32e-3

# Create dictionaries to store data:
P = {1: [], 2: [], 3: [], 4: []}
B = {1: [], 2: [], 3: [], 4: []}
C = {1: [], 2: [], 3: [], 4: []}
G = {1: [], 2: [], 3: [], 4: []}
O = {1: [], 2: [], 3: [], 4: []}
WP = {1: [], 2: [], 3: [], 4: []}
WB = {1: [], 2: [], 3: [], 4: []}
WC = {1: [], 2: [], 3: [], 4: []}
WG = {1: [], 2: [], 3: [], 4: []}
WO = {1: [], 2: [], 3: [], 4: []}
WE = {1: [], 2: [], 3: [], 4: []}
WT = {1: [], 2: [], 3: [], 4: []}

# Condense data into nested dictionaries:
fuel_names = {1 : 'Coal', 2: 'Gas', 3: 'Oil'}
fossil_fuels = {1 : C, 2: G, 3: O}
initial_fuels = {1: C0, 2: G0, 3: O0}
fuel_rates = {1: C_rate, 2: G_rate, 3: O_rate}

K = {1: 1.52e7, 2: 1.69e8, 3: 7.11e7, 4: 5e7}   # Carrying capacity of countries
WK = 10e9                                       # Carrying capacity of world
lam = 0.8                                       # Climate sensitivity

# Choose population model and simulation duration:
model = raw_input('Choose model: Malthus or Logistic? (Default Malthus): ') or 'Malthus'
T = int(raw_input('Simulation duration in years? (Default 100): ') or 100)

# Other parameters:
g = float(raw_input('Biomass growth rate? (Default 0.01): ') or 0.01)
W = 7.5e9                                       # World population

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

# Plot world population curves:
plt.clf()
for i in country:
    if model == 'Malthus':
        for t in t_axis:
            WP[i].append(WP0 * np.exp(P_rate[i] * t))
        plt.semilogy(t_axis, WP[i], label=country[i], linestyle=styles[i])
    elif model == 'Logistic':
        for t in t_axis:
            WP[i].append((WP0 * WK * np.exp(P_rate[i] * t)) / (WK + WP0 * (np.exp(P_rate[i] * t) - 1)))
        plt.semilogy(t_axis, WP[i], label=country[i], linestyle=styles[i])
    else:
        raise ValueError('Model selection not recognised.')
plt.gcf()
if model == 'Malthus':
    plt.legend(loc=2)
else:
    plt.legend(loc=4)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'World population')
plt.savefig('plots/world_population_' + model + '.pdf', bbox_inches='tight')

# Plot fossil fuel curves:
for fuel in fossil_fuels:
    plt.clf()
    for i in country:
        if model == 'Malthus':
            for t in t_axis:
                fossil_fuels[fuel][i].append(initial_fuels[fuel][i] -
                                             fuel_rates[fuel][i] * P0[i] * (np.exp(P_rate[i] * t) - 1) / P_rate[i])
            plt.semilogy(t_axis, fossil_fuels[fuel][i], label=country[i], linestyle=styles[i])
        else:
            raise NotImplementedError('Model not yet considered.')
    plt.gcf()
    plt.legend(loc=4)
    plt.xlabel(r'Time elapsed (years)')
    plt.ylabel(r'{y} (tonnes)'.format(y=fuel_names[fuel]))
    plt.savefig('plots/{y}_'.format(y=fuel_names[fuel]) + model + '.pdf', bbox_inches='tight')

# Plot biomass curves:
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
plt.savefig('plots/bio_' + model + '_g={y}.pdf'.format(y=g), bbox_inches='tight')

# Plot world coal curves:
plt.clf()
for i in country:
    if model == 'Malthus':
        for t in t_axis:
            WC[i].append(WC0 - C_rate[i] * WP0 * (np.exp(P_rate[i] * t) - 1) / P_rate[i])
        plt.semilogy(t_axis, WC[i], label=country[i], linestyle=styles[i])
    else:
        raise NotImplementedError('Model not yet considered.')
plt.gcf()
plt.legend(loc=3)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Coal (tonnes)')
plt.savefig('plots/world_coal_' + model + '.pdf', bbox_inches='tight')

# Plot world gas curves:
plt.clf()
for i in country:
    if model == 'Malthus':
        for t in t_axis:
            WG[i].append(WG0 - G_rate[i] * WP0 * (np.exp(P_rate[i] * t) - 1) / P_rate[i])
        plt.semilogy(t_axis, WG[i], label=country[i], linestyle=styles[i])
    else:
        raise NotImplementedError('Model not yet considered.')
plt.gcf()
plt.legend(loc=4)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Gas (tonnes)')
plt.savefig('plots/world_gas_' + model + '.pdf', bbox_inches='tight')

# Plot world oil curves:
plt.clf()
for i in country:
    if model == 'Malthus':
        for t in t_axis:
            WO[i].append(WO0 - O_rate[i] * WP0 * (np.exp(P_rate[i] * t) - 1) / P_rate[i])
        plt.semilogy(t_axis, WO[i], label=country[i], linestyle=styles[i])
    else:
        raise NotImplementedError('Model not yet considered.')
plt.gcf()
plt.legend(loc=3)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Oil (tonnes)')
plt.savefig('plots/world_oil_' + model + '.pdf', bbox_inches='tight')

# Plot world biomass curves:
plt.clf()
for i in country:
    if model == 'Malthus':
        for t in t_axis:
            WB[i].append(WB0 * np.exp(g * t)
                     + B_rate[i] * WP0 * (np.exp(P_rate[i] * t) - np.exp(g * t)) / (g - P_rate[i]))
        plt.semilogy(t_axis, WB[i], label=country[i], linestyle=styles[i])
    else:
        raise NotImplementedError('Model not yet considered.')
plt.gcf()
plt.legend(loc=4)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Biomass (tonnes)')
plt.savefig('plots/world_bio_' + model + '_g={y}.pdf'.format(y=g), bbox_inches='tight')

# Plot world emissions curves:
plt.clf()
for i in country:
    if model == 'Malthus':
        WE0 = WP0 * (B_CO2 * B_rate[i] + C_CO2 * C_rate[i] + G_CO2 * G_rate[i] + O_CO2 * O_rate[i]) \
                         * (np.exp(P_rate[i] * 0)) / P_rate[i]
        WE[i].append(0)
        for t in t_axis[1:]:
            WE[i].append(WP0 * (B_CO2 * B_rate[i] + C_CO2 * C_rate[i] + G_CO2 * G_rate[i] + O_CO2 * O_rate[i]) \
                         * (np.exp(P_rate[i] * t)) / P_rate[i] - WE0)
        plt.semilogy(t_axis, WE[i], label=country[i], linestyle=styles[i])
    else:
        raise NotImplementedError('Model not yet considered.')
plt.gcf()
plt.legend(loc=4)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Estimated world CO2 emissions (tonnes)')
plt.savefig('plots/C02_emissions_' + model + '_g={y}.pdf'.format(y=g), bbox_inches='tight')

# Plot world temperature change curves:
plt.clf()
for i in country:
    if model == 'Malthus':
        WT0 = lam * 5.35 * np.log((WE[i][0] / (5.3e15 + WE[i][0]) + 388e-6) / 388e-6)
        WT[i].append(0)
        for j in range(1, len(t_axis)):
            WT[i].append(lam * 5.35 * np.log((WE[i][j] /(5.3e15 + WE[i][j]) + 388e-6) / 388e-6) - WT0)
        plt.semilogy(t_axis, WT[i], label=country[i], linestyle=styles[i])
    else:
        raise NotImplementedError('Model not yet considered.')
plt.gcf()
plt.legend(loc=4)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Estimated world temperature change (Kelvin)')
plt.savefig('plots/temp_change_' + model + '_g={y}.pdf'.format(y=g), bbox_inches='tight')
