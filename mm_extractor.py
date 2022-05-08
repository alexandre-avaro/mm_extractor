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
    plt.rcParams.update({'font.size': 35})

try:
    # Read inputs (config.json file)
    with open("config.json") as f:
        config = json.load(f)

    cycle_time = float(config["cycle_time"])
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
        calibration_data = pd.read_csv("./calibration_data.csv", sep=',')
        ff_data = pd.read_csv('./flat-field.csv', sep=',')
        bg_data = pd.read_csv('./background.csv', sep=',')
        time = np.multiply([val for val in data['Cycle']
                            if not(pd.isnull(val))], cycle_time)

        # Import flat-field and background data
        ff_import = pd.read_csv('./flat-field.csv', sep=',')
        ff_data = pd.DataFrame()
        bg_import = pd.read_csv('./background.csv', sep=',')
        bg_data = pd.DataFrame()

        for k in ff_import.keys()[2:]:
            ff_data[k] = [np.median(ff_import[k])]
            bg_data[k] = [np.median(bg_import[k])]

    elif config['plate_reader'] == "AB":
        data = extract_AB(file_path)
        time = np.multiply(data['Cycle'], cycle_time)
        F_cleaved = 35100e9
        F_uncleaved = 3380e9

    # Baseline subtraction
    if baseline_subtraction:
        for plate in plate_list:
            data[plate] = data[plate]-min(data[plate])

    # Calibration curve
    signal_uncleaved_calib = []
    signal_cleaved_calib = []
    concentrations = []

    def calibration_curve(x, F, c0, a):
        return (a+F*x*10**(-x/c0))

    for i, k in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']):
        signal_uncleaved_calib.append(
            ((np.median(calibration_data[k+'1'])-bg_data[k+'1'])/(
                ff_data[k+'1']-bg_data[k+'1']))[0])
        signal_uncleaved_calib.append(
            ((np.median(calibration_data[k+'2'])-bg_data[k+'2'])/(
                ff_data[k+'2']-bg_data[k+'2']))[0])
        signal_uncleaved_calib.append(((
            np.median(calibration_data[k+'3'])-bg_data[k+'3'])/(
                ff_data[k+'3']-bg_data[k+'3']))[0])
        signal_cleaved_calib.append(((
            np.median(calibration_data[k+'4'])-bg_data[k+'4'])/(
                ff_data[k+'4']-bg_data[k+'4']))[0])
        signal_cleaved_calib.append(((
            np.median(calibration_data[k+'5'])-bg_data[k+'5'])/(
                ff_data[k+'5']-bg_data[k+'5']))[0])
        signal_cleaved_calib.append(((
            np.median(calibration_data[k+'6'])-bg_data[k+'6'])/(
                ff_data[k+'6']-bg_data[k+'6']))[0])
        concentrations.append(2**(1-i))
        concentrations.append(2**(1-i))
        concentrations.append(2**(1-i))

    params_cleaved, cov = curve_fit(calibration_curve, concentrations,
                                    signal_cleaved_calib, bounds=([
                                        0, 2, 0],
                                        [np.inf, np.inf, np.inf]))
    params_uncleaved, cov = curve_fit(calibration_curve, concentrations,
                                      signal_uncleaved_calib, bounds=([
                                          0, 2, 0],
                                          [np.inf, np.inf, np.inf]))
    F_cleaved = params_cleaved[0]
    F_uncleaved = params_uncleaved[0]
    c0 = params_uncleaved[1]

    # Michaelis-Menten "traditional" fitting (mode 1)
    if mm_fitting and not total_fitting:
        def moving_average(a, t, n=5):
            """
            Smoothing operator
            """
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return t[int(np.floor(n/2)):-int(np.floor(n/2))], ret[n - 1:] / n

        velocities = np.zeros(len(plate_list))
        for id, plate in enumerate(plate_list):
            # Calibration and smoothing
            data[plate] = (data[plate] - bg_data[plate][0])/(
                ff_data[plate][0] - bg_data[plate][0])
            data[plate] = np.multiply(10**(-substrate_concentrations[id]/c0) /
                                      (F_cleaved - F_uncleaved),
                                      (np.subtract(data[plate],
                                                   F_uncleaved *
                                                   substrate_concentrations[id]
                                                   )
                                       )
                                      )
            smooth_t, smooth_data = moving_average(np.array(data[plate]), time)
            smooth_t, smooth_data = list(smooth_t), list(smooth_data)
            smooth_data = smooth_data - np.min(smooth_data)
            grad_t, grad = moving_average(
                np.gradient(smooth_data, smooth_t[1]-smooth_t[0]),
                smooth_t)
            velocities[id] = np.max(grad)*1e-6

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
        if np.sqrt(cov[0][0])/param[0] + np.sqrt(cov[1][1])/param[1] <= 2:
            plt.plot(np.multiply(
                substrate_concentrations,
                1e6),
                velocities, 'o', fillstyle='none', markersize=14,
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
            plt.xlabel(r'$S_0$ [μM]')
            plt.ylabel(r'$V_0$ [M/s]')
            print(
                "Results of the fit (Michaelis-Menten): \n kcat = "
                + str(param[0]) + ' s^(-1) +/- ' +
                str(cov[0][0]) + ' s^(-1) \n Km = '
                + str(param[1])+' M +/- ' + str(cov[1][1]) + ' M\n' +
                str(param[0]/param[1]*1e-6)
            )
            plt.show()
        else:
            print("Michaelis-Menten fit failed.")

    # Progress curve fitting (Schnell-Mendoza solution, mode 2)
    elif total_fitting and not mm_fitting:
        # Initialize variables
        cmap = plt.get_cmap('viridis')
        kcat_list = np.zeros(len(plate_list))
        Km_list = np.zeros(len(plate_list))
        delta_kcat = np.zeros(len(plate_list))
        delta_Km = np.zeros(len(plate_list))
        t0_list = np.zeros(len(plate_list))
        new_data = {}

        fig, (ax1, ax2) = plt.subplots(1, 2)
        for id, plate in enumerate(plate_list):
            # Clean data (if all the exp time are not equal)
            new_data[plate] = [val for val in data[plate]
                               if not(pd.isnull(val))]
            S0 = substrate_concentrations[id]

            # Calibration and FF/BG
            new_data[plate] = (new_data[plate] - bg_data[plate][0])/(
                ff_data[plate][0] - bg_data[plate][0])

            new_data[plate] = np.multiply(10**(-S0/c0) /
                                          (F_cleaved - F_uncleaved),
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
                ax1.plot(time/60,
                         np.multiply(1e3, new_data[plate]), '.', markersize=2,
                         color=cmap(id/len(plate_list)))
                kcat_list[id] = param[1]
                delta_kcat[id] = np.sqrt(cov[1][1])

                Km_list[id] = param[0]
                delta_Km[id] = np.sqrt(cov[0][0])

                t0_list[id] = param[2]
                ax1.plot(time/60, np.multiply(1e3, schnell_mendoza(
                    time, param[0], param[1], param[2])),
                    color=cmap(id/len(plate_list)))
            else:
                ax1.plot(time/60,
                         np.multiply(1e3, new_data[plate]), 'x',
                         color=cmap(id/len(plate_list)))
                print("Fit for well " + plate + " failed.")
        ax1.set(xlabel=r'Time [min]')
        ax1.set(ylabel=r'Cleaved reporters [nM]')

        # Plot kcat and Km solution
        for id in range(len(plate_list)):
            if (Km_list[id]*kcat_list[id] != 0):
                ax2.plot(np.multiply(Km_list[id], 1e6), kcat_list[id], '.',
                         markersize=40, fillstyle='none',
                         markeredgewidth=2,
                         color=cmap(id/len(plate_list)))
                ax2.errorbar(np.multiply(Km_list[id], 1e6), kcat_list[id],
                             1e6*delta_Km[id], delta_kcat[id],
                             capsize=10,
                             ecolor=cmap(id/len(plate_list)))
        ax2.set(xlabel=r'$K_M$ [μM]')
        ax2.set(ylabel=r'$k_{cat}$ $[s^{-1}]$')
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.show()

    else:
        print("Only one fit expected, two or zero asked.")

except ValueError:
    print(
        "The CSV file could not be read.\
 Make sure your file is in the right format."
    )
