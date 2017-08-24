# Import required packages:
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def biomass_quadrature_step(Pop0, CarCap, r, gr, time, h):
    """
    A quadrature routine for approximating the increase in integral involved in the biomass logistic model solution.

    :param Pop0: initial human population.
    :param CarCap: carrying capacity of region considered.
    :param r: rate of growth of population.
    :param gr: biomass growth rate.
    :param time: end of time period considered.
    :param h: strip width.
    :return: integral in time interval [t-1, t].
    """
    q = 0.
    for l in range(int(time / h) - 1, int(time / h)):
        term = np.exp((r - gr) * l * h) / (CarCap + Pop0 * (np.exp(r * l * h) - 1))
        if l in (0, int(time / h) - 1):
            q += term / 2
        else:
            q += term
    return q * h


# List countries, initial populations and growth rates:
country = {1: 'Cuba', 2: 'Japan', 3: 'South Africa', 4: 'Uganda'}
P0 = {1: 11.4e6, 2: 127e6, 3: 53.3e6, 4: 37.5e6, 5: 7.5e9}          # 5th is world population
P_rate = {1: 1.39e-3, 2: 9.35e-3, 3: 1.57e-2, 4: 3.31e-2}

# List initial fuel values and consumption rates:
B0 = {1: 6.08e8, 2: 4.9e9, 3: 1.81e9, 4: 4.61e8, 5: 2.4e13}
C0 = {1: 2.44e7, 2: 5.23e8, 3: 30e9, 4: 3.49e5, 5: 4.7e12}
G0 = {1: 1.71e11, 2: 3.51e11, 3: 3.17e7, 4: 2.41e10, 5: 8.7e12}
O0 = {1: 2.43e7, 2: 3.69e8, 3: 4.53e10, 4: 4.13e8, 5: 1.9e12}
B_rate = {1: 1.39e-3, 2: 1.84e-3, 3: 1.89e-3, 4: 2.70}
C_rate = {1: 1.91e-3, 2: 1.50, 3: 2.95, 4: 0.}
G_rate = {1: 0.122, 2: 0.014, 3: 0.123, 4: 0.}
O_rate = {1: 0.11, 2: 1.54, 3: 0.72, 4: 2.09e-3}
B_CO2 = {False: 5.4e-2, True: 3.45e-3}
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
fuel_names = {1: 'Coal', 2: 'Gas', 3: 'Oil'}
fossil_fuels = {1: C, 2: G, 3: O}
world_fossil_fuels = {1: WC, 2: WG, 3: WO}
initial_fuels = {1: C0, 2: G0, 3: O0}
fuel_rates = {1: C_rate, 2: G_rate, 3: O_rate}

K = {1: 1.52e7, 2: 1.69e8, 3: 7.11e7, 4: 5e7, 5: 10e9}  # Carrying capacity of countries and world
lam = 0.8  # Climate sensitivity

# Choose population model and simulation duration:
if raw_input('Press any key other than enter to use Logistic model instead of Malthus model: '):
    model = 'Logistic'
else:
    model = 'Malthus'
T = int(raw_input('Simulation duration in years? (Default 100): ') or 100)

# Other parameters:
g = float(raw_input('Biomass growth rate? (Default 0.01): ') or 0.01)
trees = bool(raw_input('Account for tree respiration? (Default False): ') or False)
W = 7.5e9  # World population

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
    else:
        for t in t_axis:
            P[i].append((P0[i] * K[i] * np.exp(P_rate[i] * t)) / (K[i] + P0[i] * (np.exp(P_rate[i] * t) - 1)))
        plt.semilogy(t_axis, P[i], label=country[i], linestyle=styles[i])
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
            WP[i].append(P0[5] * np.exp(P_rate[i] * t))
        print 't: ', np.shape(t_axis), 'WP: ', np.shape(WP[i])
        plt.semilogy(t_axis, WP[i], label=country[i], linestyle=styles[i])
    else:
        for t in t_axis:
            WP[i].append((P0[5] * K[5] * np.exp(P_rate[i] * t)) / (K[5] + P0[5] * (np.exp(P_rate[i] * t) - 1)))
        plt.semilogy(t_axis, WP[i], label=country[i], linestyle=styles[i])
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

    # Country-wise:
    plt.clf()
    for i in country:
        for t in t_axis:
            fossil_fuels[fuel][i].append(initial_fuels[fuel][i] - fuel_rates[fuel][i]
                                         * (P[i][len(fossil_fuels[fuel][i])] - P0[i]) / P_rate[i])
        plt.semilogy(t_axis, fossil_fuels[fuel][i], label=country[i], linestyle=styles[i])
    plt.gcf()
    plt.legend(loc=4)
    plt.xlabel(r'Time elapsed (years)')
    plt.ylabel(r'{y} (tonnes)'.format(y=fuel_names[fuel]))
    plt.savefig('plots/{y1}_{y2}.pdf'.format(y1=fuel_names[fuel], y2=model), bbox_inches='tight')

    # Worldwide projection:
    plt.clf()
    for i in country:
        for t in t_axis:
            world_fossil_fuels[fuel][i].append(initial_fuels[fuel][5] - fuel_rates[fuel][i]
                                               * (WP[i][len(world_fossil_fuels[fuel][i])] - P0[5]) / P_rate[i])
        plt.semilogy(t_axis, fossil_fuels[fuel][i], label=country[i], linestyle=styles[i])
    plt.gcf()
    plt.legend(loc=4)
    plt.xlabel(r'Time elapsed (years)')
    plt.ylabel(r'{y} (tonnes)')
    plt.savefig('plots/world_{y1}_{y2}.pdf'.format(y1=fuel_names[fuel], y2=model), bbox_inches='tight')

# Plot country-wise biomass curves:
plt.clf()
for i in country:
    if model == 'Malthus':
        for t in t_axis:
            B[i].append(B0[i] * np.exp(g * t)
                        + B_rate[i] * (P[i][len(B[i])] - P0[i] * np.exp(g * t)) / (g - P_rate[i]))
        plt.semilogy(t_axis, B[i], label=country[i], linestyle=styles[i])
    else:
        I = 0
        for t in t_axis:
            I += biomass_quadrature_step(P0[i], K[i], P_rate[i], g, t, 0.1)
            B[i].append(np.exp(g * t) * (B0[i] - B_rate[i] * P0[i] * K[i] * I))
        plt.semilogy(t_axis, B[i], label=country[i], linestyle=styles[i])
plt.gcf()
plt.legend(loc=4)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Biomass (tonnes)')
plt.savefig('plots/bio_' + model + '_g={y1}.pdf'.format(y1=g), bbox_inches='tight')

# Plot worldwide projected biomass curves:
plt.clf()
for i in country:
    if model == 'Malthus':
        for t in t_axis:
            WB[i].append(B0[5] * np.exp(g * t)
                         + B_rate[i] * (WP[i][len(WB[i])] - P0[5] * np.exp(g * t)) / (g - P_rate[i]))
        plt.semilogy(t_axis, WB[i], label=country[i], linestyle=styles[i])
    else:
        I = 0
        for t in t_axis:
            I += biomass_quadrature_step(P0[5], K[5], P_rate[i], g, t, 0.1)
            WB[i].append(np.exp(g * t) * (B0[5] - B_rate[i] * P0[5] * K[5] * I))
        plt.semilogy(t_axis, B[i], label=country[i], linestyle=styles[i])
plt.gcf()
plt.legend(loc=4)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Biomass (tonnes)')
plt.savefig('plots/world_bio_' + model + '_g={y1}.pdf'.format(y1=g), bbox_inches='tight')

# Plot worldwide projected emissions curves:
plt.clf()
for i in country:
    WE0 = (B_CO2[trees] * B_rate[i] + C_CO2 * C_rate[i] + G_CO2 * G_rate[i] + O_CO2 * O_rate[i]) \
          * (WP[i][0] - P0[5]) / P_rate[i]
    WE[i].append(0)
    for t in t_axis[1:]:
        WE[i].append((B_CO2[trees] * B_rate[i] + C_CO2 * C_rate[i] + G_CO2 * G_rate[i] + O_CO2 * O_rate[i])
                     * (WP[i][len(WE[i])] - P0[5]) / P_rate[i])
    plt.semilogy(t_axis, WE[i], label=country[i], linestyle=styles[i])
plt.gcf()
plt.legend(loc=4)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Estimated world CO2 emissions (tonnes)')
plt.savefig('plots/C02_emissions_' + model + '_g={y1}_trees={y2}.pdf'.format(y1=g, y2=trees), bbox_inches='tight')

# Plot worldwide projected temperature change curves:
plt.clf()
for i in country:
    WT0 = lam * 5.35 * np.log((WE[i][0] / (5.3e15 + WE[i][0]) + 388e-6) / 388e-6)
    WT[i].append(0)
    for j in range(1, len(t_axis)):
        WT[i].append(lam * 5.35 * np.log((WE[i][j] / (5.3e15 + WE[i][j]) + 388e-6) / 388e-6) - WT0)
    plt.semilogy(t_axis, WT[i], label=country[i], linestyle=styles[i])
plt.gcf()
plt.legend(loc=4)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Estimated world temperature change (Kelvin)')
plt.savefig('plots/temp_change_{y0}_g={y1}_trees={y2}.pdf'.format(y0=model, y1=g, y2=trees), bbox_inches='tight')
