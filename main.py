"""
This is the Three or more models using OGM-GE strategy
"""
import tqdm
import time
import torch
from torch import nn
from dataset.data_loader import get_loader
from config import Config, get_config
from model import Multi_models
from utils import *


def train(model, train_loader, optimizer, criterion, scheduler, config, epoch):
    model.train()
    epoch_loss, num_batch = 0, 0
    loop = tqdm.tqdm(train_loader)
    all_a, all_v, all_t = 0., 0., 0.
    grad_a, grad_v, grad_t = 0., 0., 0.
    for idx, batch_data in enumerate(loop):
        text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data
        optimizer.zero_grad()
        with torch.cuda.device(0):
            text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                text.cuda(), visual.cuda(), audio.cuda(), y.cuda(), l.cuda(), bert_sent.cuda(), \
                bert_sent_type.cuda(), bert_sent_mask.cuda()

        pred_a, pred_v, pred_t, pred, l_ta, l_tv = model(visual, audio, bert_sent, bert_sent_type, bert_sent_mask)
        loss = criterion(pred, y) + 0.5 * l_ta + 0.5 * l_tv
        # TODO get params grad
        loss.backward()
        # TODO OGM-GE
        if config.use_OGM:
            # TODO value is less contribution is more
            score_a = criterion(pred_a, y)
            score_v = criterion(pred_v, y)
            score_t = criterion(pred_t, y)
            all_a += score_a
            all_v += score_v
            all_t += score_t
            score_all = {'audio': score_a, 'visual': score_v, 'text': score_t}
            # TODO higher—— > less
            score_all = sorted(score_all.items(), key=lambda x: x[1], reverse=True)
            score_all = dict(score_all)
            coeff_a, coeff_v, coeff_t = 0, 0, 0
            coeff = {'audio': coeff_a, 'visual': coeff_v, 'text': coeff_t}
            score_list = list(score_all.values())
            # print(score_all)
            # print(str(score_list[0]) + " " + str(score_list[1]) + " " + str(score_list[-1]))
            # print(score_all)
            # TODO reward-based OGM-GE
            for index, key in enumerate(score_all):
                 if index != 2:
                     ratio = score_list[index] / score_list[-1]
                     coeff[key] = 1 + torch.tanh(config.alpha * torch.relu(ratio))
                 else:
                     coeff[key] = 1

            #for index, key in enumerate(score_all):
            #    if index == 0:
            #        # 最大除最小
            #        # ratio = score_list[0] / score_list[index]
            #        coeff[key] = 1 # + torch.tanh(config.alpha * torch.relu(ratio))
            #    elif index == 1:
            #        ratio = score_list[0] / score_list[index]
            #        coeff[key] = 1 - torch.tanh(config.alpha * torch.relu(ratio))
            #    else:
            #        ratio = score_list[0] / score_list[index]
            #        coeff[key] = 1 - torch.tanh(config.alpha * torch.relu(ratio))
            # print(coeff)

            # for index, key in enumerate(score_all):
            #     ratio = score_list[0] / score_all[key]
            #     if ratio > 1:
            #         coeff[key] = 1 - torch.tanh(config.alpha * torch.relu(ratio))
            #     else:
            #         coeff[key] = 1

            grad_a += coeff['audio']
            grad_v += coeff['visual']
            grad_t += coeff['text']
            # print(coeff)


            for name, parms in model.named_parameters():
                if 'audio' in name and len(parms.grad.shape) != 1:
                    parms.grad = parms.grad * coeff['audio'] + \
                                 torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                if 'visual' in name and len(parms.grad.shape) != 1:
                    parms.grad = parms.grad * coeff['visual'] + \
                                 torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                if 'text' in name and parms.grad is not None:
                    if len(parms.grad.shape) != 1:
                        parms.grad = parms.grad * coeff['text'] + \
                                  torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

        optimizer.step()
        epoch_loss += loss.item()
        num_batch += 1
        loop.set_description(f'Epoch [{epoch}/{config.num_epochs}]')
        loop.set_postfix(loss=epoch_loss / num_batch)
    #scheduler.step()
    return epoch_loss / num_batch, all_a / num_batch, all_v / num_batch, all_t / num_batch, grad_a / num_batch, grad_v / num_batch, grad_t / num_batch


def evaluate(model, loader, criterion, config, test=True):
    model.eval()
    total_loss = 0.0
    result, result_a, result_v, result_t, truth = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            _, visual, _, audio, _, y, lengths, bert_sent, bert_sent_type, bert_sent_mask, _ = batch

            with torch.cuda.device(0):
                audio, visual, y, lengths = audio.cuda(), visual.cuda(), y.cuda(), lengths.cuda()
                bert_sent, bert_sent_type, bert_sent_mask = bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()

            batch_size = lengths.size(0)  # bert_sent in size (bs, seq_len, emb_size)

            # we don't need lld and bound anymore
            pred_a, pred_v, pred_t, pred, _, _ = model(visual, audio, bert_sent, bert_sent_type, bert_sent_mask)

            total_loss += criterion(pred, y).item() * batch_size

            # Collect the results into ntest if test else self.hp.n_valid)
            result.append(pred), result_a.append(pred_a), result_v.append(pred_v), result_t.append(pred_t), truth.append(y)

    avg_loss = total_loss / (config.n_test if test else config.n_valid)
    result, result_a, result_v, result_t, truth = torch.cat(result), torch.cat(result_a), torch.cat(result_v),\
                                                  torch.cat(result_t), torch.cat(truth)
    return avg_loss, result, result_a, result_v, result_t, truth


if __name__ == '__main__':
    config = Config()
    dataset = str.lower(config.dataset)
    print("Start loading the data....")
    train_config = get_config(dataset, mode='train', batch_size=config.batch_size)
    valid_config = get_config(dataset, mode='valid', batch_size=config.batch_size)
    test_config = get_config(dataset, mode='test', batch_size=config.batch_size)

    # pretrained_emb saved in train_config here
    train_loader = get_loader(train_config, shuffle=True)
    print('Training data loaded!')
    valid_loader = get_loader(valid_config, shuffle=False)
    print('Validation data loaded!')
    test_loader = get_loader(test_config, shuffle=False)
    print('Test data loaded!')
    print('Finish loading the data....')
    print('Start building model!')
    model = Multi_models(config)
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.L1Loss(reduction="mean")
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, 0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=0, factor=0.1, verbose=True)
    # TODO train and eval
    if config.use_OGM:
        print("----using OGM-GE----")
    for epoch in range(1, config.num_epochs + 1):
        start_time = time.time()
        loss, all_a, all_v, all_t, grad_a, grad_v, grad_t = train(model, train_loader, optimizer, criterion, scheduler, train_config, epoch)
        print("audio: " + str(all_a) + " visual: " + str(all_v) + " text: " + str(all_t))
        print("audio: " + str(grad_a) + " visual: " + str(grad_v) + " text: " + str(grad_t))
        train_time = time.time() - start_time
        print("----do valid and test----")
        # TODO valid
        valid_loss, _, _, _, _, _ = evaluate(model, valid_loader, criterion, valid_config, test=False)
        # TODO test
        test_loss, result, result_a, result_v, result_t, truth = evaluate(model, test_loader, criterion, test_config, test=True)
        all_time = time.time() - start_time
        print('-' * 50)
        print("%d epoch lr: %f" % (epoch, optimizer.param_groups[0]['lr']))
        print('Epoch {:2d} | Train Time {:5.4f} sec | All Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.
              format(epoch, train_time, all_time, valid_loss, test_loss))
        if dataset == 'mosei':
            print('-----All-----')
            metrics = eval_mosei_senti(result, truth, True)
            print('---Audio-----')
            eval_mosei_senti(result_a, truth, True)
            print('---Visual----')
            eval_mosei_senti(result_v, truth, True)
            print('----Text-----')
            eval_mosei_senti(result_t, truth, True)
        if dataset == 'mosi':
            print('-----All-----')
            metrics = eval_mosi(result, truth, True)
            print('---Audio-----')
            eval_mosi(result_a, truth, True)
            print('---Visual----')
            eval_mosi(result_v, truth, True)
            print('----Text-----')
            eval_mosi(result_t, truth, True)
        scheduler.step(metrics=metrics['mae'])
    torch.save(model, config.save_path)
    print('-----program end------')
