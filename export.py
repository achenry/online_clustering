import os
import pandas as pd
import shutil
import glob

customer_ids = [1000, 1001, 1002]
scenario_names = ['Online Periodic-K Crisp KMeans++', 'Online Periodic-K Fuzzy KMeans++',
                  'Offline Optimal-K Crisp KMeans++', 'Offline Optimal-K Fuzzy KMeans++',
                  'Online Optimal-K Crisp KMeans++ Window-Size 4',
                  'Online Optimal-K Fuzzy KMeans++ Window-Size 4']
sim_types = ['ie', 'cs']
plot_types = ['centres', 'params', 'metrics']

# copy images to target folder under new name
for sim_type in sim_types:
    results_table = ""
    results_path = './' + sim_type + '_results'
    export_path = './' + sim_type + '_export_results'

    if not os.path.exists(export_path):
        os.mkdir(export_path)
    else:
        files = glob.glob(export_path + '/*')
        for f in files:
            os.remove(f)

    # generate latex table of results
    final_results_df = pd.read_csv(os.path.join(results_path, 'final_results.csv'))
    for s in range(len(scenario_names)):
        if 'Offline' in scenario_names[s]:
            algo = 'okk'
        elif 'Periodic' in scenario_names[s]:
            algo = 'prk'
        elif 'Online Optimal' in scenario_names[s]:
            algo = 'efwokk'

        for c in range(len(customer_ids)):

            local_results_path = os.path.join(results_path, scenario_names[s], str(customer_ids[c]))
            if not os.path.exists(local_results_path):
                continue

            if 'Crisp' in scenario_names[s]:
                fuzziness = 'crisp'
                results_table += "Crisp " + algo.upper() + " & "
            elif 'Fuzzy' in scenario_names[s]:
                fuzziness = 'fuzzy'
                results_table += "Fuzzy " + algo.upper() + " & "
            final_results_row = final_results_df.loc[(final_results_df.CustomerID == customer_ids[c]) &
                                                     (final_results_df.ScenarioName == scenario_names[s])]

            results_table += f"${customer_ids[c]}$" \
                             f" & ${final_results_row.NumberClusters.iloc[0]}$ " \
                             f" & ${final_results_row.Compactness.iloc[0]:.3f}$" \
                             f" & ${final_results_row.DBI.iloc[0]:.3f}$" \
                             f" & ${final_results_row.Separation.iloc[0]:.3f}$" \
                             f" & ${final_results_row.Inertia.iloc[0]:.3f}$" \
                             f" & ${final_results_row.RelativeRunningTime.iloc[0]:.3f}$\\\\\n"
            for plot_type in plot_types:


                export_file = '_'.join([str(customer_ids[c]), fuzziness, algo, sim_type, plot_type])

                # get all files beginning with the filename, order based on i
                evolution_file_names = []
                evolution_file_idx = []
                for dirpath, dirnames, filenames in os.walk(local_results_path):
                    for f in filenames:
                        if plot_type in f and '.png' in f:
                            evolution_file_names.append(f)
                            evolution_file_idx.append(os.path.splitext(f)[0].split('_')[-1])

                for f, i in zip(evolution_file_names, evolution_file_idx):
                    if plot_type != 'params':
                        file_name = '_'.join([export_file, i]) + '.png'
                    else:
                        file_name = export_file + '.png'
                    shutil.copyfile(os.path.join(local_results_path, f),
                                    os.path.join(export_path, file_name))



    f = open(os.path.join(export_path, 'latex_table.txt'), 'w')
    f.write(results_table)
    f.close()
