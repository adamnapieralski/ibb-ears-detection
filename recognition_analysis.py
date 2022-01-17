import numpy as np
import json
import matplotlib.pyplot as plt

MODEL_NAMES = ['vgg16', 'resnet18', 'resnet34', 'inceptionv3', 'xception']


def plot_training_accuracies():
    figs = {
        'perfect': {
            'train_acc': plt.subplots(figsize=(7,7)),
            'val_acc': plt.subplots(figsize=(7,7))
        },
        'detected': {
            'train_acc': plt.subplots(figsize=(7,7)),
            'val_acc': plt.subplots(figsize=(7,7))
        }
    }
    for det_type in ['perfect', 'detected']:
        for m_name in MODEL_NAMES:
            logs_path = f'results/recognition/models/{m_name}_model_{det_type}_logs.csv'
            logs = np.loadtxt(logs_path, dtype=object)
            figs[det_type]['train_acc'][1].plot(logs[1:,1].astype(float), label=m_name)
            figs[det_type]['val_acc'][1].plot(logs[1:,3].astype(float), label=m_name)

    for det_type in ['perfect', 'detected']:
        for acc_type in ['train_acc', 'val_acc']:
            figs[det_type][acc_type][1].legend()
            figs[det_type][acc_type][1].set_xlabel('epoch')
            figs[det_type][acc_type][1].set_ylabel('Accuracy')
            figs[det_type][acc_type][1].set_title(
                ('Training' if acc_type == 'train_acc' else 'Validation') + f' accuracy scores - {det_type} ears data'
            )
            figs[det_type][acc_type][0].savefig(f'results/recognition/figs/fig.{acc_type}.{det_type}.png', facecolor='white')

def plot_test_accuracies():
    with open('results/recognition/evaluation_results.json', 'rt') as f:
        res = json.load(f)

    for det_type_weights in ['perfect', 'detected']:
        for det_type_eval in ['perfect', 'detected']:
            pp_1 = [res[m_name][f'{det_type_weights}_{det_type_eval}']['accuracy'] for m_name in MODEL_NAMES]
            pp_5 = [res[m_name][f'{det_type_weights}_{det_type_eval}']['accuracy_rank_5'] for m_name in MODEL_NAMES]

            fig, ax = plt.subplots()
            colors = ['cornflowerblue', 'olive', 'brown', 'purple', 'green']
            for i in range(len(MODEL_NAMES)):
                width = 0.6
                ax.bar(i, pp_5[i]*100, width = width, alpha=.5, color=colors[i])
                ax.bar(i, pp_1[i]*100, width = width, color=colors[i])
                ax.text(i - width / 2, pp_5[i]*100 - 5, '%.2f%%' % (pp_5[i]*100))
                ax.text(i - width / 2, pp_1[i]*100 - 5, '%.2f%%' % (pp_1[i]*100))

            ax.set_xticks(np.arange(len(MODEL_NAMES)), MODEL_NAMES)
            ax.set_xlabel('Model')
            ax.set_ylabel('Accuracy [%]')
            ax.set_title(f'Test accuracy scores (Rank-1, Rank-5)\nTrain: {det_type_weights} ears, Test: {det_type_eval} ears')
            fig.savefig(f'results/recognition/figs/fig.test.{det_type_weights}_{det_type_eval}.png', facecolor='white')

if __name__ == '__main__':
    plot_training_accuracies()
    plot_test_accuracies()
