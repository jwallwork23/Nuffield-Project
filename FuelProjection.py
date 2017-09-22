# Import required packages:
import numpy as np
import matplotlib

#matplotlib.use('TkAgg')
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


# List countries, initial populations and growth rates (0th entry corresponds to world population):
country = {1: 'Cuba', 2: 'Japan', 3: 'South Africa', 4: 'Uganda', 5: 'China', 6: 'India', 7: 'UK', 8: 'USA'}
P0 = {0: 7.5e9, 1: 11.4e6, 2: 127e6, 3: 53.3e6, 4: 37.5e6, 5: 1388e6, 6: 1343e6, 7: 65.5e6, 8: 327e6}
P_rate = {1: 1.39e-3, 2: 9.35e-3, 3: 1.57e-2, 4: 3.31e-2, 5: 3.9e-3, 6: 1.16e-2, 7: 6.1e-3, 8: 7.2e-3}

# List initial fuel values and consumption rates (as of 2016):
B0 = {0: 2.4e13, 1: 6.08e8, 2: 4.9e9, 3: 1.81e9, 4: 4.61e8, 5: 'TODO', 6: 'TODO', 7: 'TODO', 8: 'TODO'}
C0 = {0: 4.7e12, 1: 2.44e7, 2: 5.23e8, 3: 30e9, 4: 3.49e5, 5: 'TODO', 6: 'TODO', 7: 'TODO', 8: 'TODO'}
G0 = {0: 8.7e12, 1: 1.71e11, 2: 3.51e11, 3: 3.17e7, 4: 2.41e10, 5: 'TODO', 6: 'TODO', 7: 'TODO', 8: 'TODO'}
O0 = {0: 1.9e12, 1: 2.43e7, 2: 3.69e8, 3: 4.53e10, 4: 4.13e8, 5: 'TODO', 6: 'TODO', 7: 'TODO', 8: 'TODO'}
B_rate = {1: 1.39e-3, 2: 1.84e-3, 3: 1.89e-3, 4: 2.70, 5: 'TODO', 6: 'TODO', 7: 'TODO', 8: 'TODO'}
C_rate = {1: 1.91e-3, 2: 1.50, 3: 2.95, 4: 0., 5: 2.75e9 / P0[5], 6: 463e6 / P0[6], 7: 38e6 / P0[7], 8: 651e6 / P0[8]}
G_rate = {1: 0.122, 2: 0.014, 3: 0.123, 4: 0., 5: 177e9 * 7.70e-4, 6: 50.6e9 * 7.70e-4, 7: 68.3e9 * 7.70e-4,
          8: 778e9 * 7.70e-4}
O_rate = {1: 0.11, 2: 1.54, 3: 0.72, 4: 2.09e-3, 5: (12.0e3 * 365 * 130.5) / P0[5], 6: (4.16e3 * 365 * 130.5) / P0[6],
          7: (1.56e3 * 365 * 130.5) / P0[7], 8: (19.4e3 * 365 * 130.5) / P0[8]}
B_CO2 = {False: 5.4e-2, True: 3.45e-3}
C_CO2 = 2.30e-3
G_CO2 = 3.36e-8
O_CO2 = 3.32e-3

# Create dictionaries to store data:
P = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
B = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
C = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
G = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
O = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
WP = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
WB = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
WC = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
WG = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
WO = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
WE = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
WT = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}

# Condense data into nested dictionaries:
fuel_names = {1: 'Coal', 2: 'Gas', 3: 'Oil'}
fossil_fuels = {1: C, 2: G, 3: O}
world_fossil_fuels = {1: WC, 2: WG, 3: WO}
initial_fuels = {1: C0, 2: G0, 3: O0}
fuel_rates = {1: C_rate, 2: G_rate, 3: O_rate}

# Carrying capacity of countries and world:
K = {0: 10e9, 1: 1.52e7, 2: 1.69e8, 3: 7.11e7, 4: 5e7, 5: 1851e6, 6: 1791e6, 7: 87.3e6, 8: 436e6}
lam = 0.8  # Climate sensitivity

# Choose population model and simulation duration:
if raw_input('Press any key other than enter to use blog selection instead of Nuffield selection: '):
    selection = (5, 6, 7, 8)
else:
    selection = (1, 2, 3, 4)
choice = {}
for i in selection:
    choice[i] = country[i]
if raw_input('Press any key other than enter to use Logistic model instead of Malthus model: '):
    model = 'Logistic'
else:
    model = 'Malthus'
T = int(raw_input('Simulation duration in years? (Default 100): ') or 100)

bio = bool(raw_input('Consider biomass? (Default False): ') or False)
if bio:
    g = float(raw_input('Biomass growth rate? (Default 0.01): ') or 0.01)
    trees = bool(raw_input('Account for tree respiration? (Default False): ') or False)
else:
    g = 'OFF'
    trees = 'OFF'
cwise = bool(raw_input('Press any key other than enter to consider countrywise projections, too.'))
if raw_input('Press any key other than enter to use .png instead of .pdf: '):
    extension = '.png'
else:
    extension = '.pdf'

# Plotting setup:
styles = {1: ':', 2: '--', 3: '-.', 4: '-', 5: ':', 6: '--', 7: '-.', 8: '-'}
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('legend', fontsize='x-large')
plt.clf()
t_axis = np.linspace(0, T, T * 10)

# Plot population curves:
for i in choice:
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
plt.savefig('plots/population_' + model + extension, bbox_inches='tight')

# Plot world population curves:
plt.clf()
for i in choice:
    if model == 'Malthus':
        for t in t_axis:
            WP[i].append(P0[0] * np.exp(P_rate[i] * t))
        print 't: ', np.shape(t_axis), 'WP: ', np.shape(WP[i])
        plt.semilogy(t_axis, WP[i], label=country[i], linestyle=styles[i])
    else:
        for t in t_axis:
            WP[i].append((P0[0] * K[0] * np.exp(P_rate[i] * t)) / (K[0] + P0[0] * (np.exp(P_rate[i] * t) - 1)))
        plt.semilogy(t_axis, WP[i], label=country[i], linestyle=styles[i])
plt.gcf()
if model == 'Malthus':
    plt.legend(loc=2)
else:
    plt.legend(loc=4)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'World population')
plt.savefig('plots/world_population_' + model + extension, bbox_inches='tight')

# Plot fossil fuel curves:
for fuel in fossil_fuels:

    # Country-wise:
    if cwise:
        plt.clf()
        for i in choice:
            for t in t_axis:
                fossil_fuels[fuel][i].append(initial_fuels[fuel][i] - fuel_rates[fuel][i]
                                             * (P[i][len(fossil_fuels[fuel][i])] - P0[i]) / P_rate[i])
            plt.semilogy(t_axis, fossil_fuels[fuel][i], label=country[i], linestyle=styles[i])
        plt.gcf()
        plt.legend(loc=4)
        plt.xlabel(r'Time elapsed (years)')
        plt.ylabel(r'{y} (tonnes)'.format(y=fuel_names[fuel]))
        plt.savefig('plots/{y1}_{y2}'.format(y1=fuel_names[fuel], y2=model) + extension, bbox_inches='tight')

    # Worldwide projection:
    plt.clf()
    for i in choice:
        for t in t_axis:
            world_fossil_fuels[fuel][i].append(initial_fuels[fuel][0] - fuel_rates[fuel][i]
                                               * (WP[i][len(world_fossil_fuels[fuel][i])] - P0[0]) / P_rate[i])
        plt.semilogy(t_axis, world_fossil_fuels[fuel][i], label=country[i], linestyle=styles[i])
    plt.gcf()
    plt.legend(loc=4)
    plt.xlabel(r'Time elapsed (years)')
    plt.ylabel(r'{y} (tonnes)'.format(y=fuel_names[fuel]))
    plt.savefig('plots/world_{y1}_{y2}'.format(y1=fuel_names[fuel], y2=model) + extension, bbox_inches='tight')

# Plot country-wise biomass curves:
if bio:
    if cwise:
        plt.clf()
        for i in choice:
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
        plt.savefig('plots/bio_' + model + '_g={y1}'.format(y1=g) + extension, bbox_inches='tight')

    # Plot worldwide projected biomass curves:
    plt.clf()
    for i in choice:
        if model == 'Malthus':
            for t in t_axis:
                WB[i].append(B0[0] * np.exp(g * t)
                             + B_rate[i] * (WP[i][len(WB[i])] - P0[0] * np.exp(g * t)) / (g - P_rate[i]))
            plt.semilogy(t_axis, WB[i], label=country[i], linestyle=styles[i])
        else:
            I = 0
            for t in t_axis:
                I += biomass_quadrature_step(P0[0], K[0], P_rate[i], g, t, 0.1)
                WB[i].append(np.exp(g * t) * (B0[0] - B_rate[i] * P0[0] * K[0] * I))
            plt.semilogy(t_axis, B[i], label=country[i], linestyle=styles[i])
    plt.gcf()
    plt.legend(loc=4)
    plt.xlabel(r'Time elapsed (years)')
    plt.ylabel(r'Biomass (tonnes)')
    plt.savefig('plots/world_bio_' + model + '_g={y1}'.format(y1=g) + extension, bbox_inches='tight')

# Plot worldwide projected emissions curves:
plt.clf()
for i in choice:
    if bio:
        WE0 = (B_CO2[trees] * B_rate[i] + C_CO2 * C_rate[i] + G_CO2 * G_rate[i] + O_CO2 * O_rate[i]) \
              * (WP[i][0] - P0[0]) / P_rate[i]
    else:
        WE0 = (C_CO2 * C_rate[i] + G_CO2 * G_rate[i] + O_CO2 * O_rate[i]) * (WP[i][0] - P0[0]) / P_rate[i]
    WE[i].append(0)
    for t in t_axis[1:]:
        if bio:
            WE[i].append((B_CO2[trees] * B_rate[i] + C_CO2 * C_rate[i] + G_CO2 * G_rate[i] + O_CO2 * O_rate[i])
                         * (WP[i][len(WE[i])] - P0[0]) / P_rate[i])
        else:
            WE[i].append((C_CO2 * C_rate[i] + G_CO2 * G_rate[i] + O_CO2 * O_rate[i])
                         * (WP[i][len(WE[i])] - P0[0]) / P_rate[i])
    plt.semilogy(t_axis, WE[i], label=country[i], linestyle=styles[i])
plt.gcf()
plt.legend(loc=4)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Estimated world CO2 emissions (tonnes)')
plt.savefig('plots/C02_emissions_' + model + '_g={y1}_trees={y2}'.format(y1=g, y2=trees) + extension, bbox_inches='tight')

# Plot worldwide projected temperature change curves:
plt.clf()
for i in choice:
    WT0 = lam * 5.35 * np.log((WE[i][0] / (5.3e15 + WE[i][0]) + 388e-6) / 388e-6)
    WT[i].append(0)
    for j in range(1, len(t_axis)):
        WT[i].append(lam * 5.35 * np.log((WE[i][j] / (5.3e15 + WE[i][j]) + 388e-6) / 388e-6) - WT0)
    plt.semilogy(t_axis, WT[i], label=country[i], linestyle=styles[i])
plt.gcf()
plt.legend(loc=4)
plt.xlabel(r'Time elapsed (years)')
plt.ylabel(r'Estimated world temperature change (Kelvin)')
plt.savefig('plots/temp_change_{y0}_g={y1}_trees={y2}'.format(y0=model, y1=g, y2=trees) + extension, bbox_inches='tight')
