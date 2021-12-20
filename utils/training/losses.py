import torch

l1_loss = torch.nn.L1Loss()
l2_loss = torch.nn.MSELoss()


def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1-X)
    else:
        return torch.relu(X+1)
    
    
def compute_generator_losses(G, Y, Xt, Xt_attr, Di, embed, ZY, eye_heatmaps, loss_adv_accumulated, 
                             diff_person, same_person, args):
    # adversarial loss
    L_adv = 0.
    for di in Di:
        L_adv += hinge_loss(di[0], True).mean(dim=[1, 2, 3])
    L_adv = torch.sum(L_adv * diff_person) / (diff_person.sum() + 1e-4)

    # id loss
    L_id =(1 - torch.cosine_similarity(embed, ZY, dim=1)).mean()

    # attr loss
    if args.optim_level == "O2" or args.optim_level == "O3":
        Y_attr = G.get_attr(Y.type(torch.half))
    else:
        Y_attr = G.get_attr(Y)
    
    L_attr = 0
    for i in range(len(Xt_attr)):
        L_attr += torch.mean(torch.pow(Xt_attr[i] - Y_attr[i], 2).reshape(args.batch_size, -1), dim=1).mean()
    L_attr /= 2.0

    # reconstruction loss
    L_rec = torch.sum(0.5 * torch.mean(torch.pow(Y - Xt, 2).reshape(args.batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)
    
    # l2 eyes loss
    if args.eye_detector_loss:
        Xt_heatmap_left, Xt_heatmap_right, Y_heatmap_left, Y_heatmap_right = eye_heatmaps
        L_l2_eyes = l2_loss(Xt_heatmap_left, Y_heatmap_left) + l2_loss(Xt_heatmap_right, Y_heatmap_right)
    else:
        L_l2_eyes = 0
        
    # final loss of generator
    lossG = args.weight_adv*L_adv + args.weight_attr*L_attr + args.weight_id*L_id + args.weight_rec*L_rec + args.weight_eyes*L_l2_eyes
    loss_adv_accumulated = loss_adv_accumulated*0.98 + L_adv.item()*0.02
    
    return lossG, loss_adv_accumulated, L_adv, L_attr, L_id, L_rec, L_l2_eyes


def compute_discriminator_loss(D, Y, Xs, diff_person):
    # fake part
    fake_D = D(Y.detach())
    loss_fake = 0
    for di in fake_D:
        loss_fake += torch.sum(hinge_loss(di[0], False).mean(dim=[1, 2, 3]) * diff_person) / (diff_person.sum() + 1e-4)

    # ground truth part
    true_D = D(Xs)
    loss_true = 0
    for di in true_D:
        loss_true += torch.sum(hinge_loss(di[0], True).mean(dim=[1, 2, 3]) * diff_person) / (diff_person.sum() + 1e-4)

    lossD = 0.5*(loss_true.mean() + loss_fake.mean())

    return lossD