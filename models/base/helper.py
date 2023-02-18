# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F


def base_train(model, trainloader, optimizer, scheduler, epoch, args, is_base=False):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)

    if args.class_relation == 'None':
        class_relations = None
    else:
        if args.class_relation == 'feat':
            class_protos = model.fc.weight.detach() # [num_all_classes, c_b]
        else:
            assert args.class_relation == 'wg'
            class_protos = model.fc_base.weight.detach() # [num_all_classes, c_b]

        class_protos = class_protos[:args.base_class]
        class_protos = F.normalize(class_protos, dim=-1) # [num_classes, c_b]
        class_relations = torch.mm(class_protos, class_protos.t()) # [num_classes, num_classes]

        average_relation = (torch.sum(class_relations) - class_relations.shape[0]) / (class_relations.shape[0] * class_relations.shape[1] - class_relations.shape[0])

    if args.in_domain_feat_cls_weight != 0.0:
        if args.in_domain_class_relation == 'None':
            in_domain_class_relations = None
        else:
            if args.in_domain_class_relation == 'feat':
                in_domain_class_protos = model.in_domain_fc.weight.detach() # [num_all_classes, c_d]
            else:
                assert args.in_domain_class_relation == 'wg'
                in_domain_class_protos = model.in_domain_fc_base.weight.detach() # [num_all_classes, c_d]

            in_domain_class_protos = in_domain_class_protos[:args.base_class]
            in_domain_class_protos = F.normalize(in_domain_class_protos, dim=-1) # [num_classes, c_d]
            in_domain_class_relations = torch.mm(in_domain_class_protos, in_domain_class_protos.t()) # [num_classes, num_classes]

            in_domain_average_relation = (torch.sum(in_domain_class_relations) - in_domain_class_relations.shape[0]) / (in_domain_class_relations.shape[0] * in_domain_class_relations.shape[1] - in_domain_class_relations.shape[0])

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        logits = model(data, is_base=is_base)
        logits = logits[:, :args.base_class]
        
        train_label = train_label.to(torch.int64)
        
        if args.cosMargin != 0.0:
            label_mask = F.one_hot(train_label, args.base_class)
            if class_relations is None:
                logits = logits - label_mask * args.cosMargin * args.temperature
            else:
                label_relations = torch.mm(label_mask.float(), class_relations) # [b, num_classes]
                lower_bound = args.cosMargin
                average = args.average_cosMargin #-0.3
                adj_cosMargin = average + (lower_bound - average) / (1.0 - average_relation) * (label_relations - average_relation)
                logits = logits + (1 - label_mask) * adj_cosMargin * args.temperature  # [b, num_classes]


        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)

        if args.in_domain_feat_cls_weight != 0.0:
            backbone_feat = model.end_points['final_feature']
            in_domain_feat = model.in_domain_forward(backbone_feat)
            
            if model.in_domain_dropout_fn is None:
                in_domain_logits = F.linear(F.normalize(in_domain_feat, p=2, dim=-1), F.normalize(model.in_domain_fc.weight if is_base==False else model.in_domain_fc_base.weight, p=2, dim=-1))
            else:
                in_domain_logits = F.linear(model.in_domain_dropout_fn(F.normalize(in_domain_feat, p=2, dim=-1)), F.normalize(model.in_domain_fc.weight if is_base==False else model.in_domain_fc_base.weight, p=2, dim=-1))
            
            in_domain_logits = in_domain_logits[:, :args.base_class]
            in_domain_logits = args.temperature * in_domain_logits

            if args.in_domain_feat_cosMargin != 0.0:
                label_mask = F.one_hot(train_label, args.base_class)
                if in_domain_class_relations is None:
                    in_domain_logits = in_domain_logits - label_mask * args.in_domain_feat_cosMargin * args.temperature
                else:
                    in_domain_label_relations = torch.mm(label_mask.float(), in_domain_class_relations) # [b, num_classes]
                    in_domain_lower_bound = args.in_domain_feat_cosMargin
                    in_domain_average = args.in_domain_average_cosMargin #-0.3
                    in_domain_adj_cosMargin = in_domain_average + (in_domain_lower_bound - in_domain_average) / (1.0 - in_domain_average_relation) * (in_domain_label_relations - in_domain_average_relation)
                    in_domain_logits = in_domain_logits + (1 - label_mask) * in_domain_adj_cosMargin * args.temperature  # [b, num_classes]


            in_domain_loss = F.cross_entropy(in_domain_logits, train_label)
            in_domain_acc = count_acc(in_domain_logits, train_label)

            loss = args.backbone_feat_cls_weight * loss + args.in_domain_feat_cls_weight * in_domain_loss

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=args.num_workers, pin_memory=True, shuffle=False)
    original_transform = trainloader.dataset.transform
    trainloader.dataset.transform = transform
    embedding_list = []
    if args.in_domain_feat_cls_weight != 0.0:
        in_domain_embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.mode = 'encoder'
            embedding = model(data)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())

            if args.in_domain_feat_cls_weight != 0.0:
                backbone_feat = model.end_points['final_feature']
                in_domain_feat = model.in_domain_forward(backbone_feat)
                in_domain_embedding_list.append(in_domain_feat.cpu())

    embedding_list = torch.cat(embedding_list, dim=0)
    if args.in_domain_feat_cls_weight != 0.0:
        in_domain_embedding_list = torch.cat(in_domain_embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []
    if args.in_domain_feat_cls_weight != 0.0:
        in_domain_proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)
        
        if args.in_domain_feat_cls_weight != 0.0:
            in_domain_embedding_this = in_domain_embedding_list[data_index.squeeze(-1)]
            in_domain_embedding_this = in_domain_embedding_this.mean(0)
            in_domain_proto_list.append(in_domain_embedding_this)

    proto_list = torch.stack(proto_list, dim=0)
    model.fc.weight.data[:args.base_class] = proto_list

    if args.in_domain_feat_cls_weight != 0.0:
        in_domain_proto_list = torch.stack(in_domain_proto_list, dim=0)
        model.in_domain_fc.weight.data[:args.base_class] = in_domain_proto_list
    
    trainloader.dataset.transform = original_transform

    return model


def test(model, testloader, epoch, args, session, is_base=False):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager(); va = Averager()
    if args.in_domain_feat_cls_weight != 0.0:
        ind_va = Averager(); cmb_va = Averager()

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data, is_base=is_base)

            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label.to(torch.int64))
            acc = count_acc(logits, test_label)

            if args.in_domain_feat_cls_weight != 0.0:
                backbone_feat = model.end_points['final_feature']
                in_domain_feat = model.in_domain_forward(backbone_feat)

                in_domain_logits = F.linear(F.normalize(in_domain_feat, p=2, dim=-1), F.normalize(model.in_domain_fc.weight if is_base==False else model.in_domain_fc_base.weight, p=2, dim=-1))
                in_domain_logits = args.temperature * in_domain_logits
                in_domain_logits = in_domain_logits[:, :test_class]

                in_domain_acc = count_acc(in_domain_logits, test_label)
                combine_acc = count_acc(logits + in_domain_logits, test_label)

            vl.add(loss.item())
            va.add(acc)
            if args.in_domain_feat_cls_weight != 0.0:
                ind_va.add(in_domain_acc); cmb_va.add(combine_acc)

        vl = vl.item()
        va = va.item()
        if args.in_domain_feat_cls_weight != 0.0:
            ind_va = ind_va.item(); cmb_va = cmb_va.item()

    if args.in_domain_feat_cls_weight == 0.0:
        #print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))#; exit()
        return vl, va
    else:
        #print('epo {}, test, loss={:.4f} acc={:.4f} (ind {:.4f}, cmb {:.4f})'.format(epoch, vl, va, ind_va, cmb_va))#; exit()
        return vl, va, ind_va, cmb_va


def test_all_sessions(model, testloader, epoch, args, total_session):
    # given the input of the last session, output acc of all sessions
    model = model.eval()
    
    vls = []; vas = []; num_samples = []
    if args.in_domain_feat_cls_weight != 0.0:
        ind_vas = []; cmb_vas = []

    for i in range(total_session + 1): # the last session is for all the novel classes
        vls.append([]); vas.append([]); num_samples.append([])
        if args.in_domain_feat_cls_weight != 0.0:
            ind_vas.append([]); cmb_vas.append([])

    def SelectFromDefault(data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = torch.where(i == targets)[0]
            if data_tmp == []:
                data_tmp = data[ind_cl]
                targets_tmp = targets[ind_cl]
            else:
                data_tmp = torch.vstack((data_tmp, data[ind_cl]))
                targets_tmp = torch.hstack((targets_tmp, targets[ind_cl]))
        return data_tmp, targets_tmp

    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data)

            if args.in_domain_feat_cls_weight != 0.0:
               backbone_feat = model.end_points['final_feature']
               in_domain_feat = model.in_domain_forward(backbone_feat)

               in_domain_logits = F.linear(F.normalize(in_domain_feat, p=2, dim=-1), F.normalize(model.in_domain_fc.weight, p=2, dim=-1))
               in_domain_logits = args.temperature * in_domain_logits
           
            for k in range(total_session + 1): # the last session is for all the novel classes
                test_class = args.base_class + k * args.way
                if k < total_session:
                    test_logits = logits[:, :test_class]
                else:
                    test_logits = logits[:, args.base_class:]
                
                if k < total_session:
                    session_classes = np.arange(test_class)
                else:
                    session_classes = np.arange(args.base_class, args.base_class + (total_session-1) * args.way)

                session_logits, session_label = SelectFromDefault(test_logits, test_label, session_classes)
                
                if k == total_session:
                    session_label = session_label - args.base_class

                if len(session_label) > 0:
                    session_loss = F.cross_entropy(session_logits, session_label.to(torch.int64))
                    session_acc = count_acc(session_logits, session_label)
                    vls[k].append(session_loss); vas[k].append(session_acc); num_samples[k].append(len(session_label))

                if args.in_domain_feat_cls_weight != 0.0:
                    if k < total_session:
                        in_domain_test_logits = in_domain_logits[:, :test_class]
                    else:
                        in_domain_test_logits = in_domain_logits[:, args.base_class:]
                                       
                    in_domain_session_logits, session_label = SelectFromDefault(in_domain_test_logits, test_label, session_classes)

                    if k == total_session:
                        session_label = session_label - args.base_class

                    if len(session_label) > 0:
                        in_domain_session_acc = count_acc(in_domain_session_logits, session_label)
                        combine_session_acc = count_acc(in_domain_session_logits + session_logits, session_label)
                        ind_vas[k].append(in_domain_session_acc); cmb_vas[k].append(combine_session_acc)
        
        def get_average_results(session_vs, session_samples):
            assert len(session_vs) == len(session_samples)
            total_vs = 0.0; total_samples = 0
            for i in range(len(session_samples)):
                total_vs += session_vs[i] * session_samples[i]
                total_samples += session_samples[i]
            return total_vs / total_samples

        for i in range(total_session + 1): # the last session is for all the novel classes
            vls[i] = float(get_average_results(vls[i], num_samples[i]))
            vas[i] = float(get_average_results(vas[i], num_samples[i]))
            if args.in_domain_feat_cls_weight != 0.0:
                ind_vas[i] = float(get_average_results(ind_vas[i], num_samples[i]))
                cmb_vas[i] = float(get_average_results(cmb_vas[i], num_samples[i]))

    if args.in_domain_feat_cls_weight == 0.0:
        return vls, vas
    else:
        return vls, vas, ind_vas, cmb_vas

