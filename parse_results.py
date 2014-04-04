import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os
import re

def get_scores(root_dir):
    datas = os.listdir(root_dir)
    datas = [name for name in datas if os.path.isdir(
        root_dir + '/' + name)]

    all_scores = dict(dict(dict(dict())))

    for data in datas:
        print(data)
        targets = os.listdir(root_dir + '/' + data)

        for target in targets:
            print(target)
            files = os.listdir('%s/%s/%s/results/' % (root_dir,
                                                      data,
                                                      target))
            print(len(files))

            problem = '%s-%s' %(data, target)
            all_scores[problem] = dict(dict(dict()))

            for f in files:
                #print('%s/%s/results/%s' % (
                #    data, target, f))
                scores = pickle.load(open('%s/%s/%s/results/%s' % (
                    root_dir, data, target, f), 'rb'))

                method, cv_index, major, tmp = re.split('[-\.]', f)
                cv_index = int(cv_index)

                    
                if (major == 'rat'):
                    if (not method in all_scores[problem].keys()):
                        all_scores[problem][method] = dict(dict())

                    if (len(list(scores.keys())) > 1):
                        raise(RuntimeError("this should have only one key"))

                    scores = scores[list(scores.keys())[0]]
                        
                    for key in scores.keys():
                        if (not key in all_scores[problem][method].keys()):
                            all_scores[problem][method][key] = dict()
                            
                        all_scores[problem][method][key][cv_index] = scores[key][0]
                elif (major == 'others'):
                    for key in scores.keys():
                        if (not key in all_scores[problem].keys()):
                            all_scores[problem][key] = dict(dict())
                            all_scores[problem][key]['NA'] = dict()
                            
                        all_scores[problem][key]['NA'][cv_index] = scores[key][0][0]

    return(all_scores)

def print_log(all_scores):
    for problem in sorted(all_scores.keys()):
        print(problem)
        for method in sorted(all_scores[problem].keys()):
            print("\t", method)
            best = 0
            for parameter in sorted(all_scores[problem][method].keys()):
                values = list(all_scores[problem][method][parameter].values())
                avg = np.mean(values)
                if (avg > best):
                    message = ("\t\t%s/%d\t%g +- %g" %(parameter,
                            len(all_scores[problem][method][parameter].keys()),
                                                       avg, np.std(values)))
                    best = avg

            print(message)

def draw_plot(all_scores, problem):
    colors = ['b', 'g', 'y', 'k', 'c', 'm', 'r', '0.5', '0.9']
    index = 0
    plot_colors = []
    tmp = list()
    for method in sorted(all_scores[problem].keys()):
        print("\t", method)
        best = 0
        for parameter in sorted(all_scores[problem][method].keys()):
            tmp.append(list(all_scores[problem][method][parameter].values()))
            plot_colors.append(colors[index])
            values = list(all_scores[problem][method][parameter].values())
            avg = np.mean(values)
            if (avg > best):
                message = ("\t\t%s/%d\t%g +- %g" %(parameter,
                        len(all_scores[problem][method][parameter].keys()),
                                                   avg, np.std(values)))
                best = avg
        index += 1
        print(message)

    pl = plt.boxplot(tmp, True)
    for i in range(len(plot_colors)):
        pl['boxes'][i].set_c(plot_colors[i])
    plt.show()
    
def draw_plots(all_scores):
    for problem in sorted(all_scores.keys()):
        print(problem)
        draw_plot(all_scores, problem)
        
if __name__ == '__main__':
    root_dir = ''
    for i in range(len(sys.argv)):
        print(sys.argv[i])
        if (sys.argv[i] == '--root-dir'):
            root_dir = sys.argv[i + 1]

    if (root_dir == ''):
        root_dir = "/scratch/TL/pool0/ajalali/ratboost/data_3/"

    all_scores = get_scores(root_dir)

    print_log(all_scores)

    draw_plots(all_scores)

    draw_plot(all_scores, 'vantveer-prognosis')
