import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.optimize import curve_fit
from scipy.special import lambertw

# File selection
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    root.destroy()
    plt.rcParams.update({'font.size': 18})

try:
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
        data = pd.read_csv(file_path, sep=';')
        time = np.multiply([val for val in data[data.columns[0]]
                            if not(pd.isnull(val))], cycle_time)
    elif config['plate_reader'] == "AB":
        data = extract_AB(file_path)
        time = np.multiply(data['Cycle'], cycle_time)

    mm_fitting = bool(eval(config["michaelis_menten_fit"]))
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
    if mm_fitting:
        def calc_velocities(list, time):
            """
            Computes initial slope of the progress curve using the 4th order
            forward difference scheme.
            list : progress curve values.
            time : time array.
            """
            d = -25/12*list[0] + 4*list[1] - 3 * \
                list[2] + 4/3 * list[3] - 1/4*list[4]
            v0 = d/(time[1]-time[0])
            return float(v0)

        velocities = np.zeros(len(plate_list))
        for id, plate in enumerate(plate_list):
            # Calibration
            F_cleaved = config["F_cleaved"]
            F_uncleaved = config["F_uncleaved"]
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
            return(np.divide(np.multiply(kcatf*E0, x), (np.add(x, Kmf))))

        # Fit
        param, cov = curve_fit(
            michaelis_menten_fun,
            substrate_concentrations,
            velocities
        )

        # Plot and print results
        plt.plot(substrate_concentrations, velocities, '.')
        plt.plot(substrate_concentrations, michaelis_menten_fun(
            substrate_concentrations,
            param[0], param[1]))
        print(
            "Results of the fit (Michaelis-Menten): \n kcat = "
            + str(param[0]) + ' s^(-1)\n Km = '
            + str(param[1])+' M')
        plt.show()

    # Progress curve fitting (Schnell-Mendoza solution)
    else:
        kcat_list = np.zeros(len(plate_list))
        Km_list = np.zeros(len(plate_list))
        t0_list = np.zeros(len(plate_list))
        S0_ver = np.zeros(len(plate_list))
        new_data = {}

        fig, (ax1, ax2) = plt.subplots(1, 2)
        for id, plate in enumerate(plate_list):
            # Clean the data (if all the exp time are not equal)
            new_data[plate] = [val for val in data[plate]
                               if not(pd.isnull(val))]

            S0 = substrate_concentrations[id]
            print(S0)
            # Calibration
            F_cleaved = config["F_cleaved"]
            F_uncleaved = config["F_uncleaved"]
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
            new_data[plate] = new_data[plate][0:id_max-1]
            time = time[0:id_max-1]

            # Fit
            param, cov = curve_fit(schnell_mendoza, time, new_data[plate])

            # Eliminate failed fits and plot progress curves + fits
            if cov[0][0] + cov[1][1] <= 1e100:
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


except ValueError:
    print(
        "The CSV file could not be read.\
 Make sure your file is in the right format."
    )
