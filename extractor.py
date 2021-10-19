import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.special import lambertw

# File selection
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    root.destroy()
    plt.rcParams.update({'font.size': 15})

if True:
    # Inputs (config.json file)
    with open("config.json") as f:
        config = json.load(f)

    cycle_time = float(config["cycle_time"])

    def extract_AB(file_path):
        """
        Converts data from the Biosystems plate reader to the BioRad format
        file_path: location of the .xls AB result file
        """
        raw_data = pd.read_excel(file_path, skiprows=7)
        data = {}
        data["Cycle"] = raw_data["Cycle"]
        for k in range(len(raw_data['Well'])):
            if raw_data["Well"][k] in list(data.keys()):
                data[raw_data['Well'][k]].append(raw_data["Rn"][k])
            else:
                data[raw_data["Well"][k]] = [raw_data["Rn"][k]]
        return data

    if config['plate_reader'] == "BioRad":
        data = pd.read_csv(file_path, sep=',')
        time = np.multiply([val for val in data['Cycle']
                            if not(pd.isnull(val))], cycle_time)
        F_cleaved = 0.0032e9
        F_uncleaved = 0.00014e9

    elif config['plate_reader'] == "AB":
        data = extract_AB(file_path)
        time = np.multiply(data['Cycle'], cycle_time)
        F_cleaved = 35100e9
        F_uncleaved = 3380e9

    mm_fitting = bool(eval(config["michaelis_menten_fit"]))
    total_fitting = bool(eval(config["progress_curve_fit"]))
    baseline_subtraction = bool(
        eval(config["baseline_subtraction"]))
    plate_list = list(
        map(str, config['plate_list'].replace(" ", "").split(',')))
    E0 = 1e-9*float(config["enzyme_concentration"])
    substrate_concentrations = np.multiply(list(
        map(
            float,
            config["substrate_concentration"].split(','))), 1e-6)

    # Baseline subtraction
    if baseline_subtraction:
        for plate in plate_list:
            data[plate] = data[plate]-min(data[plate])

    # Michaelis-Menten "traditional" fitting
    if mm_fitting and not total_fitting:
        def calc_velocities(curve, time):
            """
            Computes initial slope of the progress curve using the 4th order
            forward difference scheme.
            list : progress curve values.
            time : time array.
            """
            id_right = len(curve)
            best_r = 0
            best_slope = 0
            while best_r**2 < 0.8 and id_right > 5:
                slope, intercept, r, p, se = linregress(time[0:id_right],
                                                        curve[0:id_right])
                if r**2 > best_r**2:
                    best_r, best_slope = r, slope
                id_right = id_right-1
            return max(best_slope, 0)

        velocities = np.zeros(len(plate_list))
        for id, plate in enumerate(plate_list):
            # Calibration
            data[plate] = np.multiply(1/(F_cleaved - F_uncleaved),
                                      (np.subtract(data[plate],
                                                   F_uncleaved *
                                                   substrate_concentrations[id]
                                                   )
                                       )
                                      )
            velocities[id] = calc_velocities(data[plate], time)

        def michaelis_menten_fun(x, kcatf, Kmf):
            """
            Michaelis-Menten model for v0 as a function of S0.
            x : list of substrate initial concentrations.
            kcat : kcat constant of the Michaelis-Menten model.
            Km : Km constant of the Michaelis-Menten model.
            """
            return np.divide(np.multiply(kcatf*E0, x), (np.add(x, Kmf)))

        # Fit
        param, cov = curve_fit(
            michaelis_menten_fun,
            substrate_concentrations,
            velocities
        )

        # Plot and print results
        if np.sqrt(cov[0][0])/param[0] + np.sqrt(cov[1][1])/param[1] <= 1:
            plt.plot(np.multiply(
                substrate_concentrations,
                1e6),
                velocities, '.',
                label='Exp')
            cont_substrate = np.linspace(min(substrate_concentrations),
                                         max(substrate_concentrations), 1000)
            plt.plot(np.multiply(cont_substrate, 1e6),
                     michaelis_menten_fun(
                cont_substrate,
                param[0], param[1]),
                label=r'$K_M$, $k_{cat}$')

            # Plot uncertainty kcat and Km fits
            plt.plot(np.multiply(cont_substrate, 1e6),
                     michaelis_menten_fun(
                cont_substrate,
                param[0]+cov[0][0], param[1]+cov[1][1]),
                label=r'$K_M + \sigma_{K_M}$, $k_{cat} + \sigma_{k_{cat}}$')
            plt.plot(np.multiply(cont_substrate, 1e6),
                     michaelis_menten_fun(
                cont_substrate,
                param[0]-cov[0][0], param[1]-cov[1][1]),
                label=r'$K_M - \sigma_{K_M}$, $k_{cat} - \sigma_{k_{cat}}$')
            plt.legend()
            plt.xlabel(r'$S_0$ [$\mu$M]')
            plt.ylabel(r'$V_0$ [M/s]')
            print(
                "Results of the fit (Michaelis-Menten): \n kcat = "
                + str(param[0]) + ' s^(-1)\n Km = '
                + str(param[1])+' M \n'
            )
            plt.show()
        else:
            print("Michaelis-Menten fit failed.")

    # Progress curve fitting (Schnell-Mendoza solution)
    elif total_fitting and not mm_fitting:
        kcat_list = np.zeros(len(plate_list))
        Km_list = np.zeros(len(plate_list))
        t0_list = np.zeros(len(plate_list))
        new_data = {}

        fig, (ax1, ax2) = plt.subplots(1, 2)
        for id, plate in enumerate(plate_list):
            # Clean the data (if all the exp time are not equal)
            new_data[plate] = [val for val in data[plate]
                               if not(pd.isnull(val))]

            S0 = substrate_concentrations[id]

            # Calibration
            new_data[plate] = np.multiply(1/(F_cleaved - F_uncleaved),
                                          (np.subtract(new_data[plate],
                                                       F_uncleaved*S0)
                                           )
                                          )

            def schnell_mendoza(t, Kmf, kcatf, t0):
                """
                Returns the Schnell-Mendoza solution.
                E0 : Initial enzyme concentration.
                S0 : Initial substrate concentration.
                Km : Km constant of the Michaelis-Menten model.
                kcat : kcat constant of the Michaelis-Menten model.
                t : time array.
                """
                s = Kmf*np.real(lambertw(np.multiply(S0/Kmf,
                                                     np.exp(
                                                         np.subtract(
                                                             S0/Kmf,
                                                             kcatf*E0 *
                                                             (t+t0)/Kmf))),
                                         k=0))
                c = E0*s/(np.add(Kmf, s))
                p = np.subtract(S0, np.add(c, s))
                return np.array(p)

            # Length homogeneity
            id_max = min(len(time), len(new_data[plate]))
            new_data[plate] = new_data[plate][0:id_max]
            time = time[0:id_max]

            # Fit
            param, cov = curve_fit(schnell_mendoza, time, new_data[plate])

            # Eliminate failed fits and plot progress curves + fits
            if np.sqrt(cov[0][0])/param[0] + np.sqrt(cov[1][1])/param[1] <= 1:
                ax1.plot(time,
                         np.multiply(1e6, new_data[plate]), '.', markersize=2)
                kcat_list[id] = param[1]
                Km_list[id] = param[0]
                t0_list[id] = param[2]
                ax1.plot(time, np.multiply(1e6, schnell_mendoza(
                    time, param[0], param[1], param[2])))
            else:
                ax1.plot(time,
                         np.multiply(1e6, new_data[plate]), 'x')
                print("Fit for well " + plate + " failed.")
        ax1.set(xlabel=r'$t$ [s]')
        ax1.set(ylabel=r'Cleaved reporters [$\mu$M]')
        # Plot kcat and Km solution
        ax2.plot(np.multiply(Km_list[Km_list != 0], 1e6),
                 kcat_list[kcat_list != 0], '.')
        ax2.set(xlabel=r'$K_M$ [$\mu$M]')
        ax2.set(ylabel=r'$k_{cat}$ $[s^{-1}]$')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.show()
    else:
        print("Only one fit expected, two or zero asked.")


else:
    print(
        "The CSV file could not be read.\
 Make sure your file is in the right format."
    )
