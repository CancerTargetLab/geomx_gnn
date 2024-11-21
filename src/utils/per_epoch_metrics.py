import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def epochMetrics(model_path, figure_dir, is_cs, name):
    """
    Plots metrics of model and saves them.

    Parameters:
    model_path (str): Path and name to model dict containing model metrics
    figure_dir (str): Path to figure dir
    is_cs (bool): Wether or not to set y-axis as Cosine Smilarity or Contrast Accuracy
    name (str): Title of figures
    """

    if not os.path.exists(figure_dir) and not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)

    if os.path.isdir(model_path):
        models = [torch.load(os.path.join(model_path, m_path), map_location=torch.device('cpu'), weights_only=False) for m_path in os.listdir(model_path) if m_path.endswith('.pt')]
        train_accs = [model['train_acc'] for model in models]
        val_accs = [model['val_acc'] for model in models]
        test_accs = [model['test_acc'] for model in models]
        train_losses = [model['train_list'] for model in models]
        val_losses = [model['val_list'] for model in models]
        test_losses = [model['test_list'] for model in models]

        max_len = []
        [max_len.append(len(t_accs)) for t_accs in train_accs]
        max_len = max(max_len)
        empty_array = np.zeros(max_len)
        for m in range(len(models)):
            length = len(train_accs[m])
            
            tmp_train_accs = empty_array.copy()
            tmp_train_accs[:length] = train_accs[m]
            tmp_train_accs[length:] = train_accs[m][-1]
            train_accs[m] = tmp_train_accs

            tmp_val_accs = empty_array.copy()
            tmp_val_accs[:length] = val_accs[m]
            tmp_val_accs[length:] = val_accs[m][-1]
            val_accs[m] = tmp_val_accs

            # tmp_test_accs = empty_array.copy()
            # tmp_test_accs[:length] = test_accs[m]
            # tmp_test_accs[length:] = test_accs[m][-1]
            # test_accs[m] = tmp_test_accs

            tmp_train_losses = empty_array.copy()
            tmp_train_losses[:length] = train_losses[m]
            tmp_train_losses[length:] = train_losses[m][-1]
            train_losses[m] = tmp_train_losses

            tmp_val_losses = empty_array.copy()
            tmp_val_losses[:length] = val_losses[m]
            tmp_val_losses[length:] = val_losses[m][-1]
            val_losses[m] = tmp_val_losses

            # tmp_test_losses = empty_array.copy()
            # tmp_test_losses[:length] = test_losses[m]
            # tmp_test_losses[length:] = test_losses[m][-1]
            # test_losses[m] = tmp_test_losses

        max_train_accs = np.max(train_accs, axis=0)
        min_train_accs = np.min(train_accs, axis=0)
        std_train_accs = np.std(train_accs, axis=0)
        mean_train_accs = np.mean(train_accs, axis=0)
        max_val_accs = np.max(val_accs, axis=0)
        min_val_accs = np.min(val_accs, axis=0)
        std_val_accs = np.std(val_accs, axis=0)
        mean_val_accs = np.mean(val_accs, axis=0)
        max_test_accs = np.max(test_accs, axis=0)
        min_test_accs = np.min(test_accs, axis=0)
        mean_test_accs = np.mean(test_accs, axis=0)
        max_train_losses = np.max(train_losses, axis=0)
        min_train_losses = np.min(train_losses, axis=0)
        std_train_losses = np.std(train_losses, axis=0)
        mean_train_losses = np.mean(train_losses, axis=0)
        max_val_losses = np.max(val_losses, axis=0)
        min_val_losses = np.min(val_losses, axis=0)
        std_val_losses = np.std(val_losses, axis=0)
        mean_val_losses = np.mean(val_losses, axis=0)
        max_test_losses = np.max(test_losses, axis=0)
        min_test_losses = np.min(test_losses, axis=0)
        mean_test_losses = np.mean(test_losses, axis=0)

        plt.plot(mean_train_accs, label="Train", color='red', marker='o')
        plt.plot(mean_val_accs, label='Val', color='blue', marker='o')
        plt.axhline(y=mean_test_accs, label='Test', color='orange', marker='o')
        plt.fill_between(list(range(mean_train_accs.shape[0])), max_train_accs, min_train_accs, alpha=0.3, color='red')
        plt.fill_between(list(range(mean_train_accs.shape[0])), max_val_accs, min_val_accs, alpha=0.3, color='blue')
        #plt.fill_between(list(range(mean_train_accs.shape[0])), max_test_accs, min_test_accs, alpha=0.3, color='orange')
        if is_cs:
            plt.ylabel('Cosine Similarity')
        else:
            plt.ylabel('Contrast Acc')
        plt.xlabel('Epochs')
        plt.title(name)
        plt.legend()
        if is_cs:
            plt.savefig(os.path.join(figure_dir, f'kfold_{name}_cs.png'))
        else:
            plt.savefig(os.path.join(figure_dir, f'kfold_{name}_acc.png'))
        plt.close()

        plt.plot(mean_train_losses, label="Train", color='red', marker='o')
        plt.plot(mean_val_losses, label='Val', color='blue', marker='o')
        plt.axhline(y=mean_test_losses, label='Test', color='orange', marker='o')
        plt.fill_between(list(range(mean_train_losses.shape[0])),
                         mean_train_losses-std_train_losses,
                         mean_train_losses+std_train_losses,
                         alpha=0.3,
                         color='red')
        plt.fill_between(list(range(mean_train_losses.shape[0])),
                         mean_val_losses-std_val_losses,
                         mean_val_losses+std_val_losses,
                         alpha=0.3,
                         color='blue')
        #plt.fill_between(list(range(mean_train_losses.shape[0])), max_test_losses, min_test_losses, alpha=0.3, color='orange')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title(name)
        plt.legend()
        plt.savefig(os.path.join(figure_dir, f'kfold_{name}_loss.png'))
        plt.close()

        if 'val_total_list' in models[0].keys():
            train_losses = [model['train_total_list'] for model in models]
            val_losses = [model['val_total_list'] for model in models]
            test_losses = [model['test_total_list'] for model in models]

            max_len = []
            [max_len.append(len(t_accs)) for t_accs in train_losses]
            max_len = max(max_len)
            empty_array = np.zeros(max_len)
            for m in range(len(models)):
                length = len(train_losses[m])

                tmp_train_losses = empty_array.copy()
                tmp_train_losses[:length] = train_losses[m]
                tmp_train_losses[length:] = train_losses[m][-1]
                train_losses[m] = tmp_train_losses

                tmp_val_losses = empty_array.copy()
                tmp_val_losses[:length] = val_losses[m]
                tmp_val_losses[length:] = val_losses[m][-1]
                val_losses[m] = tmp_val_losses

                # tmp_test_losses = empty_array.copy()
                # tmp_test_losses[:length] = test_losses[m]
                # tmp_test_losses[length:] = test_losses[m][-1]
                # test_losses[m] = tmp_test_losses

            max_train_losses = np.max(train_losses, axis=0)
            max_val_losses = np.max(val_losses, axis=0)
            max_test_losses = np.max(test_losses, axis=0)
            min_train_losses = np.max(train_losses, axis=0)
            min_val_losses = np.min(val_losses, axis=0)
            min_test_losses = np.min(test_losses, axis=0)
            mean_train_losses = np.mean(train_losses, axis=0)
            mean_val_losses = np.mean(val_losses, axis=0)
            mean_test_losses = np.mean(test_losses, axis=0)
            std_train_losses = np.std(train_losses, axis=0)
            std_val_losses = np.std(val_losses, axis=0)

            plt.plot(mean_train_losses, label="Train", color='red', marker='o')
            plt.plot(mean_val_losses, label='Val', color='blue', marker='o')
            plt.axhline(y=mean_test_losses, label='Test', color='orange', marker='o')
            plt.fill_between(list(range(mean_train_losses.shape[0])),
                             mean_train_losses-std_train_losses,
                             mean_train_losses+std_train_losses,
                             alpha=0.3, 
                             color='red')
            plt.fill_between(list(range(mean_train_losses.shape[0])), 
                             mean_val_losses-std_val_losses, 
                             mean_val_losses+std_val_losses, 
                             alpha=0.3, 
                             color='blue')
            #plt.fill_between(list(range(mean_train_losses.shape[0])), max_test_losses, min_test_losses, alpha=0.3, color='orange')

            plt.ylabel('Total Loss')
            plt.xlabel('Epochs')
            plt.title(name)
            plt.legend()
            plt.savefig(os.path.join(figure_dir, f'kfold_{name}_total_loss.png'))
            plt.close()
        
        if 'val_pcc_statistic_list' in models[0].keys() and len(models[0]['val_pcc_statistic_list']) > 0:
            val_pcc = [model['val_pcc_statistic_list'] for model in models]
            test_pcc = [model['test_pcc_statistic_list'] for model in models]
            val_pval = [model['val_pcc_pval_list'] for model in models]
            test_pval = [model['test_pcc_pval_list'] for model in models]

            max_len = []
            [max_len.append(len(t_accs)) for t_accs in val_pcc]
            max_len = max(max_len)
            empty_array = np.zeros(max_len)
            for m in range(len(models)):
                length = len(val_pcc[m])

                tmp_val_pcc = empty_array.copy()
                tmp_val_pcc[:length] = val_pcc[m]
                tmp_val_pcc[length:] = val_pcc[m][-1]
                val_pcc[m] = tmp_val_pcc

                # tmp_test_pcc = empty_array.copy()
                # tmp_test_pcc[:length] = test_pcc[m]
                # tmp_test_pcc[length:] = test_pcc[m][-1]
                # test_pcc[m] = tmp_test_pcc

                tmp_val_pval = empty_array.copy()
                tmp_val_pval[:length] = val_pval[m]
                tmp_val_pval[length:] = val_pval[m][-1]
                val_pval[m] = tmp_val_pval

                # tmp_test_pval = empty_array.copy()
                # tmp_test_pval[:length] = test_pval[m]
                # tmp_test_pval[length:] = test_pval[m][-1]
                # test_pval[m] = tmp_test_pval

            max_val_pcc = np.max(val_pcc, axis=0)
            max_test_pcc = np.max(test_pcc, axis=0)
            min_val_pcc = np.min(val_pcc, axis=0)
            std_val_pcc = np.std(val_pcc, axis=0)
            min_test_pcc = np.min(test_pcc, axis=0)
            mean_val_pcc = np.mean(val_pcc, axis=0)
            mean_test_pcc = np.mean(test_pcc, axis=0)
            max_val_pval = np.max(val_pval, axis=0)
            std_val_pval = np.std(val_pval, axis=0)
            max_test_pval = np.max(test_pval, axis=0)
            min_val_pval = np.min(val_pval, axis=0)
            min_test_pval = np.min(test_pval, axis=0)
            mean_val_pval = np.mean(val_pval, axis=0)
            mean_test_pval = np.mean(test_pval, axis=0)

            plt.plot(mean_val_pcc, label='Val', color='blue', marker='o')
            plt.axhline(y=mean_test_pcc, label='Test', color='orange', marker='o')
            plt.fill_between(list(range(mean_val_pcc.shape[0])), 
                             mean_val_pcc-std_val_pcc, 
                             mean_val_pcc+std_val_pcc, 
                             alpha=0.3, 
                             color='blue')
            #plt.fill_between(list(range(mean_val_pcc.shape[0])), max_test_pcc, min_test_pcc, alpha=0.3, color='orange')
            plt.ylabel('PCC')
            plt.xlabel('Epochs')
            plt.title(name)
            plt.legend()
            plt.savefig(os.path.join(figure_dir, f'kfold_{name}_pcc.png'))
            plt.close()

            plt.plot(mean_val_pval, label='Val', color='blue', marker='o')
            plt.axhline(y=mean_test_pval, label='Test', color='orange', marker='o')
            plt.fill_between(list(range(mean_val_pval.shape[0])), max_val_pval, min_val_pval, alpha=0.3, color='blue')
            #plt.fill_between(list(range(mean_val_pcc.shape[0])), max_test_pval, min_test_pval, alpha=0.3, color='orange')
            plt.ylabel('PVAL')
            plt.xlabel('Epochs')
            plt.title(name)
            plt.legend()
            plt.savefig(os.path.join(figure_dir, f'kfold_{name}_pval.png'))
            plt.close()

    else:
        model_stuff = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        train_acc = model_stuff['train_acc']
        val_acc = model_stuff['val_acc']
        train_loss = model_stuff['train_list']
        val_loss = model_stuff['val_list']

        plt.plot(train_acc, label="Train", color='red', marker='o')
        plt.plot(val_acc, label='Val', color='blue', marker='o')

        if is_cs:
            plt.ylabel('Cosine Similarity')
        else:
            plt.ylabel('Contrast Acc')
        plt.xlabel('Epochs')
        plt.title(name)
        plt.legend()
        if is_cs:
            plt.savefig(os.path.join(figure_dir, f'{name}_cs.png'))
        else:
            plt.savefig(os.path.join(figure_dir, f'{name}_acc.png'))
        plt.close()

        plt.plot(train_loss, label="Train", color='red', marker='o')
        plt.plot(val_loss, label='Val', color='blue', marker='o')

        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title(name)
        plt.legend()
        plt.savefig(os.path.join(figure_dir, f'{name}_loss.png'))
        plt.close()

        if 'val_total_list' in model_stuff.keys():
            train_loss = model_stuff['train_total_list']
            val_loss = model_stuff['val_total_list']

            plt.plot(train_loss, label="Train", color='red', marker='o')
            plt.plot(val_loss, label='Val', color='blue', marker='o')

            plt.ylabel('Total Loss')
            plt.xlabel('Epochs')
            plt.title(name)
            plt.legend()
            plt.savefig(os.path.join(figure_dir, f'{name}_total_loss.png'))
            plt.close()
        
        if 'train_ph_entropy_list' in model_stuff.keys() and len(model_stuff['train_ph_entropy_list']) > 0:
            train_loss = model_stuff['train_ph_entropy_list']
            val_loss = model_stuff['val_ph_entropy_list']

            plt.plot(train_loss, label="Train", color='red', marker='o')
            plt.plot(val_loss, label='Val', color='blue', marker='o')

            plt.ylabel('Entropy')
            plt.xlabel('Epochs')
            plt.title(name)
            plt.legend()
            plt.savefig(os.path.join(figure_dir, f'{name}_entropy.png'))
            plt.close()

        if 'train_zinb_list' in model_stuff.keys() and len(model_stuff['train_zinb_list']) > 0:
            train_loss = model_stuff['train_zinb_list']
            val_loss = model_stuff['val_zinb_list']

            plt.plot(train_loss, label="Train", color='red', marker='o')
            plt.plot(val_loss, label='Val', color='blue', marker='o')

            plt.ylabel('ZINB Loss')
            plt.xlabel('Epochs')
            plt.title(name)
            plt.legend()
            plt.savefig(os.path.join(figure_dir, f'{name}_zinbloss.png'))
            plt.close()
        
        if 'val_pcc_statistic_list' in model_stuff.keys() and len(model_stuff['val_pcc_statistic_list']) > 0:
            val_pcc = model_stuff['val_pcc_statistic_list']
            val_pval = model_stuff['val_pcc_pval_list']

            plt.plot(val_pcc, label="Val", color='red', marker='o')
            plt.ylabel('PCC')
            plt.xlabel('Epochs')
            plt.title(name)
            plt.legend()
            plt.savefig(os.path.join(figure_dir, f'{name}_pcc.png'))
            plt.close()

            plt.plot(val_pval, label='Val', color='blue', marker='o')
            plt.ylabel('PVAL')
            plt.xlabel('Epochs')
            plt.title(name)
            plt.legend()
            plt.savefig(os.path.join(figure_dir, f'{name}_pval.png'))
            plt.close()