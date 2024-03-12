import matplotlib.pyplot as plt
import torch
import os

def epochMetrics(model_path, figure_dir, is_cs, name):
    if not os.path.exists(figure_dir) and not os.path.isdir(figure_dir):
        os.makedirs(figure_dir)

    model_stuff = torch.load(model_path, map_location=torch.device('cpu'))
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